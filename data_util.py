import torch


# token_seqs and padding mask shape: (batch, max_seq_len)
def bert_mask_tokens(token_seqs, padding_mask, bert_mask_id, vocab_size):
    # Create mask to avoid padding
    seq_lens = padding_mask.sum(dim=1).unsqueeze(dim=-1)
    ignore_mask = padding_mask.clone()
    # Don't mask [CLS]
    ignore_mask[:, 0] = 0
    # Don't mask [SEP]
    ignore_mask.scatter_(dim=1, index=seq_lens - 1, src=torch.zeros_like(seq_lens).type(ignore_mask.dtype))

    # Those 15% of tokens should be predicted
    masked_tokens_mask = (torch.rand(token_seqs.shape) < 0.15) & ignore_mask.type(torch.bool)

    # We use this random draw to determine if a token should be masked (0.8), get randomly replaced (0.1) or
    # unchanged (0.1)
    p_mask_token = torch.rand(token_seqs.shape)

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

