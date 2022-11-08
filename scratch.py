from data.get_dataset import get_dataset
from torchtext.data import BucketIterator
import torch
train, val, test = get_dataset(
    root="data/",
    dataset="Yahoo",
    seq_length=100
)
loader = BucketIterator(test,  batch_size=32, sort_within_batch=True)
def batch_to_sequence(batch, device):
    sequence, sequence_len = batch.src
    sequence = sequence.to(device)
    sequence_len = sequence_len.to(device)
    return sequence, sequence_len

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_words = 0
for batch_idx, batch in enumerate(loader):
    sequence, sequence_len = batch_to_sequence(batch, device)
    total_words += torch.sub(sequence_len, 1).sum().item()  # do not take into account <SOS>
print(total_words)
print(len(train))
print(len(test))