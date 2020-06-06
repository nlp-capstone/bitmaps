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


def memory_usage_module_wise(model: nn.Module):
    total_size_bytes = 0
    for name, param in model.named_parameters():
            param_size_bytes = param.data.element_size() * param.data.nelement()

            if "cls" in name:
                continue
            if "LayerNorm" in name or "alpha" in name or "gamma" in name or "beta" in name or "position_embeddings" in name:
                total_size_bytes += param_size_bytes
                continue

            param_size_bytes //= (param.data.element_size() * 8)
            total_size_bytes += param_size_bytes
    return total_size_bytes


def count_correct(logits, truth, mask):
    return (logits.argmax(dim=-1) == truth)[mask].sum()


def mean_reciprocal_ranking(logits, truth, mask):
    top_k_ids = logits.topk(k=32, dim=-1)[1]
    truth_in_top_k_one_hot = (top_k_ids == truth.unsqueeze(dim=-1)).type(torch.int8)
    truth_in_top_k = truth_in_top_k_one_hot.sum(dim=-1)
    truth_in_top_k_idx = truth_in_top_k_one_hot.argmax(dim=-1).type(logits.dtype) + 1
    truth_in_top_k_idx[~truth_in_top_k.type(torch.bool)] = float("inf")
    return (1 / truth_in_top_k_idx[mask]).sum()

    # args = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
    # ranks = args.gather(dim=-1, index=truth.unsqueeze(-1)) + 1
    # mask_ranks = 1 / ranks[mask].type(logits.dtype)
    # return mask_ranks.sum()


def mlm_cross_entropy(logits, truth, mask):
    flat_logits = logits.view(-1, logits.shape[-1])
    flat_truth = truth.flatten()

    ce = F.cross_entropy(flat_logits, flat_truth, reduction="none") \
        .reshape(mask.shape[0], -1)

    # probs = flat_logits.softmax(dim=-1).gather(dim=-1, index=flat_truth.view(-1, 1))[mask.flatten()]

    return ce[mask].sum()
