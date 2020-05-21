import os
import glob
from multiprocessing import Pool
from functools import partial

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from data_util import bert_mask_tokens


class ShardedBertPretrainingDataset(Dataset):
    def __init__(self, data, tokenizer, random_seed=None):
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer

        self.rng_state = None
        if random_seed is None:
            return

        with torch.random.fork_rng():
            torch.random.manual_seed(random_seed)
            self.rng_state = torch.random.get_rng_state()

    @staticmethod
    def create(dir, tokenizer: PreTrainedTokenizer, max_seq_len, splits, num_processes=4, random_seed=None):
        cache_dir = os.path.join(dir, f"cache_{max_seq_len}")
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        datasets = []
        for name, subset_size in splits:
            data_path = os.path.join(cache_dir, f"data_{name}_{subset_size}.pt")
            if os.path.isfile(data_path):
                data = torch.load(data_path)
                assert data.shape[0] == subset_size and data.shape[1] == max_seq_len
                datasets.append(ShardedBertPretrainingDataset(data, tokenizer, random_seed))

        if len(datasets) == len(splits):
            return datasets[0] if len(datasets) == 1 else datasets
        elif len(datasets) > 0:
            assert False, "Need to have either all datasets cached or none to avoid overlap"

        # CLS/SEP tokens
        effective_max_seq_len = max_seq_len - len(tokenizer.build_inputs_with_special_tokens([]))

        sequences = []

        shard_paths = sorted(glob.glob(os.path.join(dir, "*.txt")), key=ShardedBertPretrainingDataset.extract_shard_index)

        # func = partial(ShardedBertPretrainingDataset.process_shard, tokenizer, effective_max_seq_len, max_seq_len, cache_dir)
        # with Pool(processes=num_processes) as pool:
        #     for t in pool.imap(func, shard_paths):
        #         sequences.append(t)

        for shard_path in shard_paths:
            sequences.append(ShardedBertPretrainingDataset.process_shard(tokenizer, effective_max_seq_len,
                                                                         max_seq_len, cache_dir, shard_path))

        data = torch.cat(sequences, dim=0)

        set_seed = random_seed is not None
        with torch.random.fork_rng(enabled=set_seed):
            if set_seed:
                torch.random.manual_seed(random_seed)
            permutation = torch.randperm(len(data))

        cur_subset_idx = 0
        for name, subset_size in splits:
            assert cur_subset_idx + subset_size <= len(data)
            subset_indices = permutation[cur_subset_idx:cur_subset_idx+subset_size].clone()
            subset_data = data[subset_indices].clone()

            torch.save(subset_indices, os.path.join(cache_dir, f"subset_indices_{name}_{subset_size}.pt"))
            torch.save(subset_data, os.path.join(cache_dir, f"data_{name}_{subset_size}.pt"))

            datasets.append(ShardedBertPretrainingDataset(subset_data, tokenizer, random_seed))
            cur_subset_idx += subset_size

        return datasets[0] if len(datasets) == 1 else datasets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        custom_rng = self.rng_state is not None
        with torch.random.fork_rng(enabled=custom_rng):
            if custom_rng:
                torch.random.set_rng_state(self.rng_state)

            sequence = self.data[idx].unsqueeze(0)
            padding_mask = sequence != self.tokenizer.pad_token_id
            masked_token_seq, masked_tokens_mask = bert_mask_tokens(sequence, padding_mask,
                                                                    self.tokenizer.mask_token_id,
                                                                    self.tokenizer.vocab_size)
            if custom_rng:
                self.rng_state = torch.random.get_rng_state()

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