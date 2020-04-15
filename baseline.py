import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertForMaskedLM.from_pretrained("bert-large-uncased")

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

sentence = "I am taking a [MASK] on Natural Language Processing."

input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(dim=0)
logits = model(input_ids)[0][0].detach()
probabilities = logits.softmax(dim=-1)

mask_mask = input_ids[0] == tokenizer.vocab["[MASK]"]
for i, is_mask in enumerate(mask_mask):
    if not is_mask:
        continue
    top_k, top_k_indices = probabilities[i].topk(k=4)
    print(i, tokenizer.convert_ids_to_tokens(top_k_indices))

