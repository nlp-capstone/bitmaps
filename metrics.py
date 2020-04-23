import torch
import torch.nn.functional as F
import torch.nn as nn


def memory_usage(model: nn.Module):
    total_size_bytes = 0
    for param in model.parameters():
        param_size_bytes = param.data.element_size() * param.data.nelement()

        # Efficiently check if there are only 1 or 2 distinct elements by check if all elements
        # equal the min or max. This is much faster than counting unique elements.
        min_w = param.data.min()
        max_w = param.data.max()
        if ((param.data == min_w) | (param == max_w)).all():
            param_size_bytes //= (param.data.element_size() * 8)

        total_size_bytes += param_size_bytes
    return total_size_bytes


def count_correct(logits, truth, mask):
    return (logits.argmax(dim=-1) == truth)[mask].sum()


def mean_reciprocal_ranking(logits, truth, mask):
    args = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
    ranks = args.gather(dim=-1, index=truth.unsqueeze(-1)) + 1
    mask_ranks = 1 / ranks[mask].type(logits.dtype)
    return torch.sum(mask_ranks.sum())


def mlm_cross_entropy(logits, truth, mask):
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_truth = truth.flatten()

    ce = F.cross_entropy(flat_logits, flat_truth, reduction="none") \
        .reshape(mask.shape[0], -1)

    return ce[mask].sum()
