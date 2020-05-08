from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import BertTokenizer

from metrics import *
from data_util import *

from BERT.binary import BinaryBert
from BERT.original import OriginalBert

torch.set_printoptions(sci_mode=False)


device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

teacher_model = OriginalBert.from_pretrained("bert-base-uncased", output_hidden_states=True).to(device).half()
model = BinaryBert.from_pretrained("bert-base-uncased", output_hidden_states=True).to(device)

train_dataset = ShardedBertPretrainingDataset("/home/tobiasr/Documents/bitmaps/pretraining_data/train/", tokenizer,
                                              128, random_seed=481, subset_size=65536)
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

epochs = 32
knowledge_transfer = False
model.transfer_lambda = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

model.train()
# teacher_model.eval()
optimizer = optim.Adam(model.parameters(), lr=1e-6, eps=1e-6)

for epoch in range(epochs):
    epoch_loss = 0.0
    print(f"Epoch {epoch + 1}/{epochs}")
    tqdm_dataloader = tqdm(dataloader)
    num_masked_tokens = 0
    for batch in tqdm_dataloader:
        masked_token_seqs, masked_tokens_mask, sequences, padding_mask = (t.to(device) for t in batch)

        num_batch_masked_tokens = masked_tokens_mask.sum().item()
        num_masked_tokens += num_batch_masked_tokens

        num_masked = masked_tokens_mask.sum().item()

        mlm_labels = sequences.clone()
        mlm_labels[~masked_tokens_mask] = -100

        # teacher_hidden_states = None
        if knowledge_transfer:
            with torch.no_grad():
                _, teacher_hidden_states = teacher_model(input_ids=masked_token_seqs, attention_mask=padding_mask)

        loss, logits, *_ = model(input_ids=masked_token_seqs,
                                 attention_mask=padding_mask,
                                 masked_lm_labels=mlm_labels,
                                 teacher_hidden_states=None)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * num_batch_masked_tokens
        tqdm_dataloader.set_description(f"Loss - {epoch_loss / num_masked_tokens} ")

    print("Loss: ", epoch_loss / num_masked_tokens)

