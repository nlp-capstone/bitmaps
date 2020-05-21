from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import BertTokenizer

from metrics import *
from datasets.cola_dataset import CoLADataset, collate_fn
from evaluate import evaluate
from apex import amp

from BERT.binary import BinaryBertForClassification, BinaryBert
from BERT.original import OriginalBert
from lr_linear_decay import LRLinearDecay

import os

torch.set_printoptions(sci_mode=False)


BASE_LOG_PATH = "..."

device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BinaryBert.from_pretrained("bert-base-uncased").to(device)

train_dataset = CoLADataset("...", tokenizer)

batch_size = 32
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=lambda samples: collate_fn(samples, tokenizer.pad_token_id))

epochs = 64

optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.1)
total_steps = epochs * len(dataloader) // batch_size
lr_decay = LRLinearDecay(int(total_steps * 0.06), total_steps)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_decay.get_lr_fn())

EXPERIMENT_NAME = "..."

log_path = os.path.join(BASE_LOG_PATH, EXPERIMENT_NAME)

cp = torch.load(os.path.join(log_path, "..."))
model.load_state_dict(cp["model_state_dict"], strict=False)

model, optimizer = amp.initialize(model, optimizer, opt_level="O2", max_loss_scale=2048.0)

print(memory_usage_module_wise(model) // (2 ** 20))

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    model.train()

    seqs = 0
    epoch_loss = 0.0
    tqdm_dataloader = tqdm(dataloader)

    tp = tn = fp = fn = 0

    for batch_index, batch in enumerate(tqdm_dataloader, start=1):
        sequences, padding_mask, labels = (t.to(device) for t in batch)

        loss, logits = model(input_ids=sequences, attention_mask=padding_mask, labels=labels)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        seqs += len(sequences)

        predicted_labels = logits.argmax(dim=-1, keepdim=True)

        cur_tp = (torch.logical_and(predicted_labels == labels, predicted_labels == 1)).sum().item()
        cur_tn = (torch.logical_and(predicted_labels == labels, predicted_labels == 0)).sum().item()
        cur_fp = (torch.logical_and(predicted_labels != labels, predicted_labels == 1)).sum().item()
        cur_fn = (torch.logical_and(predicted_labels != labels, predicted_labels == 0)).sum().item()

        assert cur_tp + cur_tn + cur_fp + cur_fn == len(labels)

        tp += cur_tp
        tn += cur_tn
        fp += cur_fp
        fn += cur_fn

        p = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if p == 0.:
            p = 1.

        mcc = (tp * tn - fp * fn) / (p ** 0.5)

        epoch_loss += loss.item() * len(sequences)
        tqdm_dataloader.set_description(f"Loss - {epoch_loss / seqs} Accuracy - {(tp + tn) / seqs} MCC - {mcc} TP - {tp} TN - {tn} FP - {fp} FN - {fn}")