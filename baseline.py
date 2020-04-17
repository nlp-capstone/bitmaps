import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
import torchtext
from torchtext import data
from tqdm import tqdm


def batch_indices(data_len, batch_size):
    for i in range(0, data_len, batch_size):
        yield i, i + batch_size


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

with torch.no_grad():
    for name, param in model.named_parameters():
        #  o(x) = [ 1, if x > 0
        #         [ 0, otherwise
        # mask = param.data > 0.0
        # param.data[mask] = 1.0
        # param.data[~mask] = 0.0
        #  o(x) = [ 1, if x >= 0
        #         [ -1, otherwise
        mask = param.data > 0.0
        param.data[mask] = 1.0
        param.data[~mask] = -1.0

# sentence = "I am taking a {} on Natural Language Processing.".format(tokenizer.mask_token)
#
# input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(dim=0)
# logits = model(input_ids)[0][0].detach()
# probabilities = logits.softmax(dim=-1)

# mask_mask = input_ids[0] == tokenizer.mask_token_id
# for i, is_mask in enumerate(mask_mask):
#     if not is_mask:
#         continue
#     top_k, top_k_indices = probabilities[i].topk(k=4)
#     print(i, tokenizer.convert_ids_to_tokens(top_k_indices))

# TODO: figure out batching
data_field = data.Field(lower=True, tokenize=lambda s: [s])
train_ptb, val_ptb, test_ptb = torchtext.datasets.PennTreebank.splits(data_field)

data = next(train_ptb.__iter__()).text
# Use BERT's [UNK] token and remove <eos> instances
# TODO: Clean up data using sanitize_ptb_tokenized_string from AllenNLP
data = [d.replace("<unk>", tokenizer.unk_token) for d in data if d != "<eos>"]
encoded_tokens = tokenizer.batch_encode_plus(data, return_tensors="pt", pad_to_max_length=True)
token_seqs = encoded_tokens["input_ids"]
masked_tokens = token_seqs.clone()

# Create mask to avoid padding
padding_mask = encoded_tokens["attention_mask"]
seq_lens = padding_mask.sum(dim=1).unsqueeze(dim=-1)
# Don't mask [CLS]
padding_mask[:, 0] = 0
# Don't mask [SEP]
padding_mask.scatter_(dim=1, index=seq_lens - 1, src=torch.zeros_like(seq_lens).type(padding_mask.dtype))

# TODO: also 0.8 MASK, 0.1 random token, 0.1 original token
masked_tokens_mask = (torch.rand(token_seqs.shape) < 0.15) & padding_mask.type(torch.bool)

# random_tokens = torch.randint_like(token_seqs, 999, len(tokenizer.vocab))
masked_tokens[masked_tokens_mask] = tokenizer.mask_token_id  # random_tokens[masked_tokens_mask]

device = torch.device("cuda")
model = model.to(device)

total_ce = 0
total_ranks = 0
batch_size = 64
correct = 0

with torch.no_grad():
    for batch_idx in tqdm(batch_indices(len(masked_tokens), batch_size),
                          total=len(masked_tokens) // batch_size + (len(masked_tokens) % batch_size != 0)):
        start, end = batch_idx
        batch_masked = masked_tokens[start:end]
        batch_mask = masked_tokens_mask[start:end]
        truth = token_seqs[start:end]
        logits = model(batch_masked.to(device))[0]

        correct += (logits.argmax(dim=-1) == truth.to(device))[batch_mask].sum()

        torch.cuda.empty_cache()

        # Calculate CE
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_truth = truth.flatten()

        ce = F.cross_entropy(flat_logits, flat_truth.to(device), reduction="none") \
              .reshape(batch_masked.shape[0], -1)

        total_ce += ce[batch_mask].sum()

        torch.cuda.empty_cache()

        # Calculate MRR
        args = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
        ranks = args.gather(dim=-1, index=truth.unsqueeze(-1).to(device)) + 1
        mask_ranks = 1 / ranks[batch_mask].type(logits.dtype)
        total_ranks += torch.sum(mask_ranks.sum())

        torch.cuda.empty_cache()

total_masked = masked_tokens_mask.sum()
print("CE : ", total_ce / total_masked)
print("Acc: ", correct / total_masked)
print("MRR: ", total_ranks / total_masked)
