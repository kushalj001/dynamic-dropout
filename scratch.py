from data.get_dataset import get_dataset

train, val, test = get_dataset(
    root="data/",
    dataset="Yahoo",
    seq_length=100
)

print(len(train.fields))
print(train.fields)
print(train.examples[0].__dict__)
print(len(train.examples))