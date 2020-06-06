from tqdm import tqdm
import os
import glob

from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import BertTokenizer

from metrics import *
from datasets.sharded_bert_pretraining_dataset import ShardedBertPretrainingDataset
from evaluate import evaluate
from apex import amp

from BERT.binary import BinaryBert
from lr_linear_decay import LRLinearDecay

torch.set_printoptions(sci_mode=False)
import matplotlib.pyplot as plt
import numpy as np

def print_log_stats(title, log_path, last_n=0):
    print("Experiment: ", os.path.basename(log_path))
    state_files = glob.glob(os.path.join(log_path, "*.pt"))
    state_files.sort(key=lambda path: int(os.path.basename(path).split("_")[2]))
    train_losses = []
    val_losses = []
    for state_file in state_files[-last_n:]:
        state_dict = torch.load(state_file, map_location="cpu")
        print(f"| Epoch {state_dict['epoch']:02d} | Loss - {state_dict['loss']:.4f} | MRR - {state_dict['mrr']:.4f} | Acc - {state_dict['acc']:.4f} |")
        train_losses.append(state_dict["loss"])
        val_losses.append(state_dict["val_loss"])
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title(title)
    plt.legend()
    plt.show()

BASE_LOG_PATH = "..."

device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BinaryBert.from_pretrained("bert-base-uncased").to(device)

train_dataset, val_dataset = ShardedBertPretrainingDataset.create("...", tokenizer,
                                                                  128, [("train", 65536, None), ("val", 8192, 481)])

# test_dataset = ShardedBertPretrainingDataset.create("...", tokenizer,
#                                              128, [("test", 8192, 481 * 481)],)

dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

epochs = 64
optimizer = optim.Adam(model.parameters(), lr=5e-4, eps=1e-6)
lr_decay = LRLinearDecay(8 * 512, 64 * 512)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_decay.get_lr_fn())

EXPERIMENT_NAME = "alpha_NO_gamma_binary_embeddings_VOCAB_NEW_lr_5e-4_grad_norm"
log_path = os.path.join(BASE_LOG_PATH, EXPERIMENT_NAME)

# print_log_stats(log_path, 0)

if not os.path.isdir(log_path):
    os.mkdir(log_path)

start_epoch = 0

# # VAL DATA FIX
# state_files = glob.glob(os.path.join(log_path, "*.pt"))
# state_files.sort(key=lambda path: int(os.path.basename(path).split("_")[2]))

# for state_file in state_files:
#     state_dict = torch.load(state_file)
#     model.load_state_dict(state_dict["model_state_dict"], strict=False)
#     print("Epoch:", state_dict["epoch"])
#     print(evaluate(model, val_dataset, device))
#
# quit()

# cp = torch.load(os.path.join(log_path, "cp_epoch_63_loss_2.6369_val_loss_2.9853.pt"))
# model.load_state_dict(cp["model_state_dict"], strict=True)
# #optimizer.load_state_dict(cp["optimizer_state_dict"])
# if "lr_scheduler" in cp:
#     lr_scheduler.load_state_dict(cp["lr_scheduler"])
# start_epoch = cp["epoch"]
# print("Starting loss: ", cp["loss"])

# Init params
for m in model.modules():
    if hasattr(m, "init_binary_weights"):
        m.init_binary_weights()

model, optimizer = amp.initialize(model, optimizer, opt_level="O2", max_loss_scale=1024.0)

print(memory_usage_module_wise(model) / (2 ** 20))

for epoch in range(start_epoch, epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    model.train()

    epoch_loss = 0.0
    epoch_mlm_loss = 0.0
    epoch_transfer_loss = 0.0
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

        mlm_loss, transfer_loss, logits = model(input_ids=masked_token_seqs,
                                                attention_mask=padding_mask,
                                                masked_lm_labels=mlm_labels)

        loss = mlm_loss + transfer_loss

        mrr += mean_reciprocal_ranking(logits, sequences, masked_tokens_mask).item()
        correct += count_correct(logits, sequences, masked_tokens_mask).item()

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)

        optimizer.step()

        epoch_loss += loss.item() * num_batch_masked_tokens
        epoch_mlm_loss += mlm_loss.item() * num_batch_masked_tokens
        epoch_transfer_loss += transfer_loss.item() * num_batch_masked_tokens
        lr = lr_scheduler.get_lr()
        tqdm_dataloader.set_description(
            f"Loss - {epoch_loss / num_masked_tokens} [MLM - {epoch_mlm_loss / num_masked_tokens}, TL - {epoch_transfer_loss / num_masked_tokens}] MRR32 - {mrr / num_masked_tokens} Acc - {correct / num_masked_tokens} Lr - {lr}")

        lr_scheduler.step()

    val_metrics = evaluate(model, val_dataset, device)

    print(val_metrics)

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "loss": epoch_loss / num_masked_tokens,
        "transfer_loss": epoch_transfer_loss / num_masked_tokens,
        "mrr": mrr / num_masked_tokens,
        "acc": correct / num_masked_tokens,
        # Val metrics
        "val_loss": val_metrics["Loss"],
        "val_mrr": val_metrics["MRR32"],
        "val_acc": val_metrics["Accuracy"]
        },
        os.path.join(log_path, f"cp_epoch_{epoch + 1}_loss_{epoch_loss / num_masked_tokens:.4f}_val_loss_{val_metrics['Loss']:.4f}.pt"))

