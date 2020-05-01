import torchtext
from torchtext import data
from tqdm import tqdm

from metrics import *
from data_util import *

from transformers import BertTokenizer
import torch

from BERT.binary import BinaryBert
from BERT.original import OriginalBert


import torch.optim as optim

torch.set_printoptions(sci_mode=False)

device = torch.device("cuda")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

teacher_model = OriginalBert.from_pretrained("bert-base-uncased", output_hidden_states=True)
model = BinaryBert.from_pretrained("bert-base-uncased", output_hidden_states=True)

data_field = data.Field(lower=True, tokenize=lambda s: [s])
train_ptb, val_ptb, test_ptb = torchtext.datasets.PennTreebank.splits(data_field)

# ---------------- Prepare training data ----------------
data = next(train_ptb.__iter__()).text

data = [d for d in [clean_sentence(d, tokenizer.unk_token) for d in data] if d is not None][:8192]
max_seq_len = 64
encoded_tokens = tokenizer.batch_encode_plus(data,
                                             add_special_tokens=True,
                                             return_tensors="pt",
                                             max_length=max_seq_len,
                                             pad_to_max_length=True)
token_seqs = encoded_tokens["input_ids"].to(device)
padding_mask = encoded_tokens["attention_mask"].to(device)

masked_tokens, masked_tokens_mask =\
    bert_mask_tokens(token_seqs, padding_mask, tokenizer.mask_token_id, len(tokenizer.vocab), device)


model = model.to(device)
teacher_model = teacher_model.to(device)

masked_tokens = masked_tokens.to(device)
masked_tokens_mask = masked_tokens_mask.to(device)

# ---------------- Training ----------------
batch_size = 64
epochs = 16

model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-6, eps=1e-6)
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_mrr = 0.0
    total_masked = 0
    total_batches = len(list(batch_indices(len(masked_tokens), batch_size)))

    # Permute data
    perm = torch.randperm(masked_tokens.shape[0])
    masked_tokens = masked_tokens[perm]
    masked_tokens_mask = masked_tokens_mask[perm]
    token_seqs = token_seqs[perm]

    for batch_idx in tqdm(batch_indices(len(masked_tokens), batch_size),
                          total=len(masked_tokens) // batch_size + (len(masked_tokens) % batch_size != 0)):
        start, end = batch_idx
        batch_masked_tokens = masked_tokens[start:end]
        batch_padding_mask = padding_mask[start:end]
        batch_mask = masked_tokens_mask[start:end]
        truth = token_seqs[start:end]

        mlm_labels = truth.clone()
        mlm_labels[~batch_mask] = -100

        with torch.no_grad():
            _, teacher_hidden_states = teacher_model(input_ids=batch_masked_tokens, attention_mask=batch_padding_mask)
        loss, logits, *_ = model(input_ids=batch_masked_tokens,
                                 attention_mask=batch_padding_mask,
                                 masked_lm_labels=mlm_labels,
                                 teacher_hidden_states=teacher_hidden_states)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_loss += loss.item()
            total_masked += torch.sum(batch_mask).item()
            epoch_acc += count_correct(logits, truth, batch_mask).item()
            epoch_mrr += mean_reciprocal_ranking(logits, truth, batch_mask).item()
    print()
    print(f"=== Epoch {epoch+1}/{epochs} ===")
    print(f"Loss: {epoch_loss / total_batches}")
    print(f"Acc : {epoch_acc / total_masked}")
    print(f"Mrr : {epoch_mrr / total_masked}")
    print()

# ---------------- Prepare validation data ----------------
data = next(val_ptb.__iter__()).text

data = [d for d in [clean_sentence(d, tokenizer.unk_token) for d in data] if d is not None]
max_seq_len = 64
encoded_tokens = tokenizer.batch_encode_plus(data,
                                             add_special_tokens=True,
                                             return_tensors="pt",
                                             max_length=max_seq_len,
                                             pad_to_max_length=True)
token_seqs = encoded_tokens["input_ids"].to(device)
padding_mask = encoded_tokens["attention_mask"].to(device)

masked_tokens, masked_tokens_mask =\
    bert_mask_tokens(token_seqs, padding_mask, tokenizer.mask_token_id, len(tokenizer.vocab), device)


model = model.to(device)
teacher_model = teacher_model.to(device)

masked_tokens = masked_tokens.to(device)
masked_tokens_mask = masked_tokens_mask.to(device)

# ---------------- Evaluation ----------------
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

total_time = 0
total_ce = 0
total_ranks = 0
correct = 0

model.eval()
with torch.no_grad():
    # Warm up cuda
    model(input_ids=torch.full((batch_size, max_seq_len), tokenizer.pad_token_id,
                               device=device,
                               dtype=masked_tokens.dtype))

    for batch_idx in tqdm(batch_indices(len(masked_tokens), batch_size),
                          total=len(masked_tokens) // batch_size + (len(masked_tokens) % batch_size != 0)):
        start, end = batch_idx
        batch_masked_tokens = masked_tokens[start:end]
        batch_padding_mask = padding_mask[start:end]
        batch_mask = masked_tokens_mask[start:end]
        truth = token_seqs[start:end]

        mlm_labels = truth.clone()
        mlm_labels[~batch_mask] = -100

        start_event.record()
        loss, logits, *_ = model(input_ids=batch_masked_tokens,
                                 attention_mask=batch_padding_mask,
                                 masked_lm_labels=mlm_labels)
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()

        total_time += start_event.elapsed_time(end_event)

        # For accuracy
        correct += count_correct(logits, truth, batch_mask).item()

        # Calculate Cross entropy, we could use their loss, but might as well use our implementation
        total_ce += mlm_cross_entropy(logits, truth, batch_mask).item()

        # Calculate MRR
        total_ranks += mean_reciprocal_ranking(logits, truth, batch_mask).item()


total_masked = masked_tokens_mask.sum().item()
print(f"Time       : {total_time / len(masked_tokens)}ms")
print("CE         : ", total_ce / total_masked)
print("Perplexity :", 2 ** (total_ce / total_masked))
print("Acc        : ", correct / total_masked)
print("MRR32      : ", total_ranks / total_masked)
print(f"Mem        : {memory_usage(model) // (2 ** 20)}MB")


