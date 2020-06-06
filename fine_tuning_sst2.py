from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import BertTokenizer

from metrics import *
from datasets.cola_dataset import collate_fn
from datasets.sst2_dataset import SST2Dataset
from apex import amp

from BERT.binary import BinaryBertForBinaryClassification
from lr_linear_decay import LRLinearDecay

import os

torch.set_printoptions(sci_mode=False)


BASE_LOG_PATH = "..."

device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BinaryBertForBinaryClassification.from_pretrained("bert-base-uncased").to(device)

print("Loading data")
train_dataset = SST2Dataset(".../SST-2/train.tsv", tokenizer)
val_dataset = SST2Dataset(".../SST-2/dev.tsv", tokenizer)

batch_size = 32
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=lambda samples: collate_fn(samples, tokenizer.pad_token_id))
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True, collate_fn=lambda samples: collate_fn(samples, tokenizer.pad_token_id))

epochs = 20

optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-6, weight_decay=0.1)
batches = len(dataloader)
lr_decay = LRLinearDecay(1 * batches, 10 * batches)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_decay.get_lr_fn())

EXPERIMENT_NAME = "binary_embeddings_h_sst2_from_bin_pretrained"

log_path = os.path.join(BASE_LOG_PATH, EXPERIMENT_NAME)

if not os.path.isdir(log_path):
    os.mkdir(log_path)

print("Loading model")
cp = torch.load(os.path.join(".../SST2/binary_embeddings_h_sst2_from_bin_pretrained/", "BINARY_cp_epoch_10_loss_0.5488_val_loss_0.5267.pt"))
model.load_state_dict(cp["model_state_dict"])
optimizer.load_state_dict(cp["optimizer_state_dict"])
# lr_scheduler.load_state_dict(cp["lr_scheduler"])
model.zero_grad()
start_epoch = 10

# Init params
# for m in model.modules():
#     if hasattr(m, "init_binary_weights"):
#         m.init_binary_weights()

model, optimizer = amp.initialize(model, optimizer, opt_level="O2", max_loss_scale=2048.0)

print(memory_usage_module_wise(model) // (2 ** 20))

for epoch in range(start_epoch, epochs, 1):
    print(f"Epoch {epoch + 1}/{epochs}")

    model.train()

    epoch_correct = 0
    seqs = 0
    epoch_loss = 0.0
    tqdm_dataloader = tqdm(dataloader)

    model.train()
    for batch_index, batch in enumerate(tqdm_dataloader, start=1):
        sequences, padding_mask, labels = (t.to(device) for t in batch)

        loss, logits = model(input_ids=sequences, attention_mask=padding_mask, labels=labels)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
        # loss.backward()
        optimizer.step()
        lr_scheduler.step()
        seqs += len(sequences)

        predicted_labels = (logits >= 0).type_as(labels)
        epoch_correct += (predicted_labels == labels).long().sum().item()

        epoch_loss += loss.item() * len(sequences)
        tqdm_dataloader.set_description(f"Loss - {epoch_loss / seqs:.4f} Accuracy - {epoch_correct / seqs:.4f}")

    # Validate
    tqdm_val_dataloader = tqdm(val_dataloader)

    val_correct = 0
    val_seqs = 0
    val_loss = 0.

    model.eval()
    for batch_index, batch in enumerate(tqdm_val_dataloader, start=1):
        sequences, padding_mask, labels = (t.to(device) for t in batch)

        with torch.no_grad():
            loss, logits = model(input_ids=sequences, attention_mask=padding_mask, labels=labels)

            val_seqs += len(sequences)

            predicted_labels = (logits >= 0).type_as(labels)
            val_correct += (predicted_labels == labels).long().sum().item()

            val_loss += loss.item() * len(sequences)
            tqdm_val_dataloader.set_description(f"Loss - {val_loss / val_seqs:.4f} Accuracy - {val_correct / val_seqs:.4f}")

    print(f"Loss - {val_loss / val_seqs:.4f} Accuracy - {val_correct / val_seqs:.4f}")

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "loss": epoch_loss / seqs,
        "acc": epoch_correct / seqs,
        # Val metrics
        "val_loss": val_loss / val_seqs,
        "val_acc": val_correct / val_seqs
        },
        os.path.join(log_path, f"BINARY_cp_epoch_{epoch + 1}_loss_{epoch_loss / seqs:.4f}_val_loss_{val_loss / val_seqs:.4f}.pt"))
