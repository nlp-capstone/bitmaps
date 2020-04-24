import torch


# https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py, not published yet
def sanitize_ptb_tokenized_string(text: str) -> str:
    """
    Sanitizes string that was tokenized using PTBTokenizer
    """
    tokens = text.split(" ")
    if len(tokens) == 0:
        return text

    # Replace quotation marks and parentheses
    token_map = {
        "``": '"',
        "''": '"',
        "-lrb-": "(",
        "-rrb-": ")",
        "-lsb-": "[",
        "-rsb-": "]",
        "-lcb-": "{",
        "-rcb-": "}",
        "<s>": "",
        "</s>": "",
    }

    # Merge punctuation with previous tokens
    punct_forward = {"`", "$", "#"}
    punct_backward = {".", ",", "!", "?", ":", ";", "%", "'"}

    # Exact matches that get merged forward or backward
    em_forward = {"(", "[", "{"}
    em_backward = {"n't", "na", ")", "]", "}"}

    new_tokens = []

    merge_fwd = False
    for i, orig_token in enumerate(tokens):
        tokens[i] = token_map[orig_token.lower()] if orig_token.lower() in token_map else orig_token
        new_token = tokens[i].lower()

        # merge_fwd was set by previous token, so it should be prepended to current token
        if merge_fwd:
            tokens[i] = tokens[i - 1] + tokens[i]

        if len(tokens[i]) == 0:
            continue

        # Special cases for `` and '', those tells us if " is the start or end of a quotation.
        # Also always merge tokens starting with ' backward and don't merge back if we just merged forward
        merge_bckwd = not merge_fwd and (
            orig_token == "''"
            or new_token in em_backward
            or new_token.startswith("'")
            or all(c in punct_backward for c in new_token)
        )
        merge_fwd = (
            orig_token == "``"
            or new_token in em_forward
            or all(c in punct_forward for c in new_token)
        )

        if merge_bckwd and new_tokens:
            new_tokens[-1] += tokens[i]
        elif not new_tokens or not merge_fwd or i == len(tokens) - 1:
            new_tokens.append(tokens[i])

    return " ".join(new_tokens)


def clean_sentence(s: str, unk_token):
    if not s or s == "<eos>":
        return None
    s = s.strip().replace("<unk>", unk_token)
    return sanitize_ptb_tokenized_string(s)


# token_seqs and padding mask shape: (batch, max_seq_len)
def bert_mask_tokens(token_seqs, padding_mask, bert_mask_id, vocab_size, device):
    torch.manual_seed(481)

    # Create mask to avoid padding
    seq_lens = padding_mask.sum(dim=1).unsqueeze(dim=-1)
    ignore_mask = padding_mask.clone()
    # Don't mask [CLS]
    ignore_mask[:, 0] = 0
    # Don't mask [SEP]
    ignore_mask.scatter_(dim=1, index=seq_lens - 1, src=torch.zeros_like(seq_lens).type(ignore_mask.dtype))

    # Those 15% of tokens should be predicted
    masked_tokens_mask = (torch.rand(token_seqs.shape, device=device) < 0.15) & ignore_mask.type(torch.bool)

    # We use this random draw to determine if a token should be masked (0.8), get randomly replaced (0.1) or
    # unchanged (0.1)
    p_mask_token = torch.rand(token_seqs.shape, device=device)

    masked_token_seqs = token_seqs.clone()

    # 80% get set to mask token
    mask_token_mask = (p_mask_token < 0.8) & masked_tokens_mask
    masked_token_seqs[mask_token_mask] = bert_mask_id

    # 10% get set to random token
    rand_token_mask = (p_mask_token >= 0.9) & masked_tokens_mask
    random_tokens = torch.randint_like(token_seqs, 0, vocab_size - 1)
    masked_token_seqs[rand_token_mask] = random_tokens[rand_token_mask]

    # 10% stay the same, this happens implicitly

    return masked_token_seqs, masked_tokens_mask


def batch_indices(data_len, batch_size):
    for i in range(0, data_len, batch_size):
        yield i, i + batch_size
