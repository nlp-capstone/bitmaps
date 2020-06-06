import torch
from torch.utils.data import Dataset
import csv

from transformers import PreTrainedTokenizer


class SST2Dataset(Dataset):
    def __init__(self, path, tokenizer: PreTrainedTokenizer):
        super().__init__()

        self.sentences = []
        self.labels = []

        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)  # Ignore header
            for row in reader:
                # Each row contains a sentence and label (either 0 or 1)
                sentence, label = row
                seq = tokenizer.encode(sentence.lower().strip(), add_special_tokens=True, return_tensors="pt")
                self.sentences.append(seq)
                self.labels.append(torch.tensor([int(label)]))

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index].unsqueeze(0)

    def __len__(self):
        return len(self.sentences)
