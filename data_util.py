import os
import glob
from multiprocessing import Pool
from functools import partial

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer


class ShardedBertPretrainingDataset(Dataset):
    def __init__(self, dir, tokenizer: PreTrainedTokenizer, max_seq_len, num_processes=4, subset_size=None,
                 random_seed=None):
        self.tokenizer = tokenizer
        cache_dir = os.path.join(dir, f"cache_{max_seq_len}")
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        if random_seed is not None:
            torch.random.manual_seed(random_seed)

        if subset_size is not None:
            data_path = os.path.join(cache_dir, f"data_{subset_size}.pt")
            if os.path.isfile(data_path):
                data = torch.load(data_path)
                assert data.shape[0] == subset_size and data.shape[1] == max_seq_len
                self.data = data
                return

        # CLS/SEP tokens
        effective_max_seq_len = max_seq_len - len(tokenizer.build_inputs_with_special_tokens([]))

        sequences = []

        shard_paths = sorted(glob.glob(os.path.join(dir, "*.txt")), key=self.extract_shard_index)

        # func = partial(self.process_shard, tokenizer, effective_max_seq_len, max_seq_len, cache_dir)
        # with Pool(processes=8) as pool:
        #     for t in pool.imap(func, shard_paths):
        #         sequences.append(t)
        for shard_path in shard_paths:
            sequences.append(self.process_shard(tokenizer, effective_max_seq_len, max_seq_len, cache_dir,
                                                shard_path))

        print("Concatenating shards")
        self.data = torch.cat(sequences, dim=0)
        print("Done concatenating shards")

        if subset_size is not None:
            subset_indices = torch.randperm(len(self.data))[:subset_size].clone()
            self.data = self.data[subset_indices].clone()

            torch.save(subset_indices, os.path.join(cache_dir, f"subset_indices_{subset_size}.pt"))
            torch.save(self.data, os.path.join(cache_dir, f"data_{subset_size}.pt"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx].unsqueeze(0)
        padding_mask = sequence != self.tokenizer.pad_token_id
        masked_token_seq, masked_tokens_mask = bert_mask_tokens(sequence, padding_mask,
                                                                self.tokenizer.mask_token_id,
                                                                self.tokenizer.vocab_size)

        return (masked_token_seq.squeeze(0), masked_tokens_mask.squeeze(0), sequence.squeeze(0),
                padding_mask.squeeze(0))

    @staticmethod
    def extract_shard_index(shard_path):
        return int(os.path.splitext(shard_path)[0].split("_")[-1])

    @staticmethod
    def process_shard(tokenizer, effective_max_seq_len, max_seq_len, cache_dir, shard_path):
        print("Processing shard #", ShardedBertPretrainingDataset.extract_shard_index(shard_path))

        # Try loading tensor for shard from disk if exists
        shard_cache_path = os.path.join(cache_dir, os.path.splitext(os.path.basename(shard_path))[0] + ".pt")
        if os.path.isfile(shard_cache_path):
            print("Using cached shard from ", shard_cache_path)
            try:
                seq_tensor = torch.load(shard_cache_path)
                assert seq_tensor.shape[1] == max_seq_len
                return seq_tensor
            except:
                os.remove(shard_cache_path)

        sequence_tensors = []
        current_sequence = []
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                # Skip empty lines
                if not line:
                    continue
                tokens = tokenizer.tokenize(line)

                # We can't encode those
                if len(tokens) > effective_max_seq_len:
                    tokens = []

                # If the current line doesn't fit into the current sequence, finalize sequence
                # by encoding as ids
                if len(tokens) + len(current_sequence) > effective_max_seq_len:
                    seq_tensor = tokenizer.encode(current_sequence, add_special_tokens=True,
                                                  max_length=max_seq_len, return_tensors="pt",
                                                  pad_to_max_length=True)
                    sequence_tensors.append(seq_tensor)
                    current_sequence.clear()
                current_sequence += tokens

        # Concatenate along batch dimension
        seq_tensor = torch.cat(sequence_tensors, dim=0)

        # Cache tensor
        try:
            torch.save(seq_tensor, shard_cache_path)
        except:
            pass
        return seq_tensor


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


def batch_indices(data_len, batch_size):
    for i in range(0, data_len, batch_size):
        yield i, i + batch_size
