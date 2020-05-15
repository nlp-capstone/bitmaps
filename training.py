from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import BertTokenizer

from metrics import *
from data_util import *
from evaluate import evaluate
from apex import amp

from BERT.binary import BinaryBert
from BERT.original import OriginalBert
from lr_linear_decay import LRLinearDecay

torch.set_printoptions(sci_mode=False)


def print_log_stats(log_path, last_n=0):
    print("Experiment: ", os.path.basename(log_path))
    state_files = glob.glob(os.path.join(log_path, "*.pt"))
    state_files.sort(key=lambda path: int(os.path.basename(path).split("_")[2]))
    for state_file in state_files[-last_n:]:
        state_dict = torch.load(state_file, map_location="cpu")
        print(f"| Epoch {state_dict['epoch']:02d} | Loss - {state_dict['loss']:.4f} | MRR - {state_dict['mrr']:.4f} | Acc - {state_dict['acc']:.4f} |")
    print()


BASE_LOG_PATH = "..."

device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

teacher_model = OriginalBert.from_pretrained("bert-base-uncased", output_attentions=True).to(device)
teacher_model.eval()
model = BinaryBert.from_pretrained("bert-base-uncased",
                                   output_attentions=True,
                                   attention_probs_dropout_prob=0.05,
                                   hidden_dropout_prob=0.05).to(device)

train_dataset = ShardedBertPretrainingDataset("...", tokenizer,
                                              128, random_seed=None, subset_size=65536)

# test_dataset = ShardedBertPretrainingDataset("...", tokenizer,
#                                              128, random_seed=481, subset_size=8192)

dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)


epochs = 64
knowledge_transfer = False
model.transfer_lambda = [0.] * 12
optimizer = optim.Adam(model.parameters(), lr=5e-4, eps=1e-6)
lr_decay = LRLinearDecay(8 * 512, 64 * 512)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_decay.get_lr_fn())

EXPERIMENT_NAME = "..."

log_path = os.path.join(BASE_LOG_PATH, EXPERIMENT_NAME)

# print_log_stats(log_path, 0)

if not os.path.isdir(log_path):
    os.mkdir(log_path)

start_epoch = 0

cp = torch.load(os.path.join(log_path, "...pt"))
model.load_state_dict(cp["model_state_dict"])
optimizer.load_state_dict(cp["optimizer_state_dict"])
if "lr_scheduler" in cp:
    lr_scheduler.load_state_dict(cp["lr_scheduler"])
start_epoch = cp["epoch"]
print("Starting loss: ", cp["loss"])

# w = 0.00975384097546339, b = 0.1711723357439041
# model.transfer_lambda = 1 + (torch.arange(12).float() - 11.) * 0.00975384097546339

# Init params
# for m in model.modules():
#     if hasattr(m, "init_params"):
#         m.init_params()

model, optimizer = amp.initialize(model, optimizer, opt_level="O2", max_loss_scale=1024.0)
model.train()

print(memory_usage_module_wise(model) // (2 ** 20))

# for p in optimizer.param_groups:
#     p["lr"] = 1e-4

for epoch in range(start_epoch, epochs):
    epoch_loss = 0.0
    epoch_mlm_loss = 0.0
    epoch_transfer_loss = 0.0
    print(f"Epoch {epoch + 1}/{epochs}")
    tqdm_dataloader = tqdm(dataloader)
    num_masked_tokens = 0
    mrr = 0
    correct = 0

    for batch_index, batch in enumerate(tqdm_dataloader, start=1):
        masked_token_seqs, masked_tokens_mask, sequences, padding_mask = (t.to(device) for t in batch)

        num_batch_masked_tokens = masked_tokens_mask.sum().item()
        num_masked_tokens += num_batch_masked_tokens

        num_masked = masked_tokens_mask.sum().item()

        mlm_labels = sequences.clone()
        mlm_labels[~masked_tokens_mask] = -100

        teacher_attentions = None
        if knowledge_transfer:
            with torch.no_grad():
                _, teacher_attentions = teacher_model(input_ids=masked_token_seqs, attention_mask=padding_mask)
                teacher_attentions = [h.float() for h in teacher_attentions]

        mlm_loss, transfer_loss, logits, attentions = model(input_ids=masked_token_seqs,
                                                            attention_mask=padding_mask,
                                                            masked_lm_labels=mlm_labels)

        # seq_lens = padding_mask.sum(dim=-1)
        # attention_loss = 0.
        #
        # batch_head_mask = masked_tokens_mask.unsqueeze(dim=1).expand(attentions[0].shape[:-1])
        # seq_lens = seq_lens.view(-1, 1, 1).expand(attentions[0].shape[:-1])[batch_head_mask]
        #
        # for l in range(len(attentions)):
        #     s_attn = attentions[l][batch_head_mask]
        #     t_attn = teacher_attentions[l][batch_head_mask]
        #
        #     s_attn[s_attn == 0.] = 1.
        #     t_attn[t_attn == 0.] = 1.
        #
        #     # attention_loss += ((s_attn - t_attn) ** 2).sum(dim=-1).mean()
        #
        #     mean_kl_divs = ((s_attn * torch.log(s_attn / t_attn)).sum(dim=1) / seq_lens)
        #     attention_loss += mean_kl_divs.mean()
        # transfer_loss = attention_loss / len(attentions)

        loss = mlm_loss + transfer_loss

        mrr += mean_reciprocal_ranking(logits, sequences, masked_tokens_mask).item()
        correct += count_correct(logits, sequences, masked_tokens_mask).item()

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * num_batch_masked_tokens
        epoch_mlm_loss += mlm_loss.item() * num_batch_masked_tokens
        epoch_transfer_loss += transfer_loss.item() * num_batch_masked_tokens
        lr = lr_scheduler.get_lr()
        tqdm_dataloader.set_description(
            f"Loss - {epoch_loss / num_masked_tokens} [MLM - {epoch_mlm_loss / num_masked_tokens}, TL - {epoch_transfer_loss / num_masked_tokens}] MRR32 - {mrr / num_masked_tokens} Acc - {correct / num_masked_tokens} Lr - {lr}")

        lr_scheduler.step()

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "loss": epoch_loss / num_masked_tokens,
        "transfer_loss": epoch_transfer_loss / num_masked_tokens,
        "mrr": mrr / num_masked_tokens,
        "acc": correct / num_masked_tokens},
        os.path.join(log_path, f"cp_epoch_{epoch + 1}_loss_{epoch_loss / num_masked_tokens:.4f}.pt"))

