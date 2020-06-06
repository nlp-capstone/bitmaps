import torch
from torch.utils.data import Dataset
import csv

from transformers import PreTrainedTokenizer


class CoLADataset(Dataset):
    def __init__(self, path, tokenizer: PreTrainedTokenizer):
        super().__init__()

        self.sentences = []
        self.labels = []

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                assert len(row) == 4
                code, label, original_label, sentence = row
                seq = tokenizer.encode(sentence.lower(), add_special_tokens=True, return_tensors="pt")
                self.sentences.append(seq)
                self.labels.append(torch.tensor([int(label)]))

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index].unsqueeze(0)

    def __len__(self):
        return len(self.sentences)


def collate_fn(samples, pad_token):
    largest_seq = max(sample[0].shape[1] for sample in samples)

    batch_seqs = torch.full((len(samples), largest_seq), fill_value=pad_token, dtype=samples[0][0].dtype)

    for i, sample in enumerate(samples):
        seq = sample[0]
        batch_seqs[i, :seq.shape[1]] = seq

    batch_labels = torch.cat([seq[1] for seq in samples], dim=0)

    return batch_seqs, batch_seqs != pad_token, batch_labels

