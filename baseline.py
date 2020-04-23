import torch
import torch.nn.functional as F
import torchtext
from torchtext import data
from tqdm import tqdm

from metrics import *
from data_util import *

from transformers import BertTokenizer
import torch

from BERT.bert_mlm import BertForMaskedLM

device = torch.device("cuda")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# with torch.no_grad():
#     for name, param in model.named_parameters():
#         #  o(x) = [ 1, if x > 0
#         #         [ 0, otherwise
#         # mask = param.data > 0.0
#         # param.data[mask] = 1.0
#         # param.data[~mask] = 0.0
#         #  o(x) = [ 1, if x >= 0
#         #         [ -1, otherwise
#         mask = param.data > 0.0
#         param.data[mask] = 1.0
#         param.data[~mask] = -1.0

data_field = data.Field(lower=True, tokenize=lambda s: [s])
train_ptb, val_ptb, test_ptb = torchtext.datasets.PennTreebank.splits(data_field)

data = next(val_ptb.__iter__()).text

data = [d for d in [clean_sentence(d, tokenizer.unk_token) for d in data] if d is not None]

encoded_tokens = tokenizer.batch_encode_plus(data, add_special_tokens=True, return_tensors="pt", pad_to_max_length=True)
token_seqs = encoded_tokens["input_ids"].to(device)
padding_mask = encoded_tokens["attention_mask"].to(device)

masked_tokens, masked_tokens_mask =\
    bert_mask_tokens(token_seqs, padding_mask, tokenizer.mask_token_id, len(tokenizer.vocab), device)


model = model.to(device)
model.eval()

masked_tokens = masked_tokens.to(device)
masked_tokens_mask = masked_tokens_mask.to(device)

total_ce = 0
total_ranks = 0
batch_size = 128
correct = 0

with torch.no_grad():
    for batch_idx in tqdm(batch_indices(len(masked_tokens), batch_size),
                          total=len(masked_tokens) // batch_size + (len(masked_tokens) % batch_size != 0)):
        start, end = batch_idx
        batch_masked_tokens = masked_tokens[start:end]
        batch_padding_mask = padding_mask[start:end]
        batch_mask = masked_tokens_mask[start:end]
        truth = token_seqs[start:end]

        mlm_labels = truth.clone()
        mlm_labels[~batch_mask] = -100
        loss, logits, *_ = model(input_ids=batch_masked_tokens,
                                 attention_mask=batch_padding_mask,
                                 masked_lm_labels=mlm_labels)

        # For accuracy
        correct += count_correct(logits, truth, batch_mask).item()

        # Calculate Cross entropy, we could use their loss, but might as well use our implementation
        total_ce += mlm_cross_entropy(logits, truth, batch_mask).item()

        # Calculate MRR
        total_ranks += mean_reciprocal_ranking(logits, truth, batch_mask).item()

total_masked = masked_tokens_mask.sum().item()
print("CE : ", total_ce / total_masked)
print("Acc: ", correct / total_masked)
print("MRR: ", total_ranks / total_masked)
print(f"Mem: {memory_usage(model) // (2 ** 20)}MB")


