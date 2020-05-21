from tqdm import tqdm

from torch.utils.data import DataLoader

from transformers import BertTokenizer

from metrics import *


def evaluate(model, dataset, device):
    # Need to copy model because we convert to half precision (not anymore for now)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # For keeping track of metrics throughout evaluation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_time = 0
    total_ce = 0
    total_ranks = 0
    total_masked = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Transfer all tensors to GPU and unpack
            masked_token_seqs, masked_tokens_mask, sequences, padding_mask = (t.to(device) for t in batch)

            num_masked = masked_tokens_mask.sum().item()

            mlm_labels = sequences.clone()
            mlm_labels[~masked_tokens_mask] = -100

            start_event.record()
            mlm_loss, transfer_loss, logits, *_ = model(input_ids=masked_token_seqs,
                                                        attention_mask=padding_mask,
                                                        masked_lm_labels=mlm_labels)
            torch.cuda.synchronize()
            end_event.record()
            torch.cuda.synchronize()

            total_time += start_event.elapsed_time(end_event)

            # For accuracy
            correct += count_correct(logits, sequences, masked_tokens_mask).item()

            # Calculate Cross entropy and scale it back up
            total_ce += (mlm_loss + transfer_loss).item() * num_masked

            # Calculate MRR
            total_ranks += mean_reciprocal_ranking(logits, sequences, masked_tokens_mask).item()

            total_masked += num_masked

    return {
        "Time(ms)": total_time / len(dataset),
        "Loss": total_ce / total_masked,
        "Perplexity": 2 ** (total_ce / total_masked),
        "Accuracy": correct / total_masked,
        "MRR32": total_ranks / total_masked,
        "Memory(MB)": memory_usage_module_wise(model) // (2 ** 20)
    }
#
# from BERT.original import OriginalBert
# device = torch.device("cuda:0")
# model = OriginalBert.from_pretrained("bert-base-uncased")
# print(evaluate(model, device))
