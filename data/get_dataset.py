import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchtext.datasets import TranslationDataset
from torchtext.data import Field


def _dummy_transform(x):
    return x


def get_bouncing_balls(root, seq_length, transform):

    # path to data set
    datapath = os.path.join(root, "bballs")
    if not os.path.exists(datapath):
        print("You need to generate the data first using provided script.")
        exit()

    # create dataset objects for pytorch
    print("Creating data set objects")
    trainset = TensorDataset(
        transform(torch.from_numpy(np.load(root+"/bballs/bb_train_data.npy").astype('float32')[:, :seq_length])),
        torch.from_numpy(np.load(root+"/bballs/bb_train_labels.npy"))
    )
    validset = TensorDataset(
        transform(torch.from_numpy(np.load(root+"/bballs/bb_valid_data.npy").astype('float32')[:, :seq_length])),
        torch.from_numpy(np.load(root+"/bballs/bb_valid_labels.npy"))
    )
    testset = TensorDataset(
        transform(torch.from_numpy(np.load(root+"/bballs/bb_test_data.npy").astype('float32')[:, :seq_length])),
        torch.from_numpy(np.load(root+"/bballs/bb_test_labels.npy"))
    )

    # output some stats
    print("Data set objects created")
    print("Len of trainset", len(trainset))
    print("Len of validset", len(validset))
    print("Len of testset", len(testset))

    return trainset, validset, testset


def get_kth(root, seq_length, transform):

    name = "kth"
    # path to data set
    datapath = os.path.join(root, name)
    if not os.path.exists(datapath):
        print("You need to generate the data first using provided script.")
        exit()

    # create dataset objects for pytorch
    print("Creating data set objects")
    trainset = TensorDataset(
        transform(torch.from_numpy(np.load(root+"/"+name+"/kth_train_data.npy").astype('float32')[:, :seq_length])),
        torch.from_numpy(np.load(root+"/"+name+"/kth_train_labels.npy"))
    )
    validset = TensorDataset(
        transform(torch.from_numpy(np.load(root+"/"+name+"/kth_valid_data.npy").astype('float32')[:, :seq_length])),
        torch.from_numpy(np.load(root+"/"+name+"/kth_valid_labels.npy"))
    )

    # output some stats
    print("Data set objects created")
    print("Len of trainset", len(trainset))
    print("Len of validset", len(validset))

    return trainset, validset, None


def _get_textfield():
    def _tokenize_en(text):
        return [x for x in text.split(" ") if x != "" and x.find(" ") == -1]
    return Field(tokenize=_tokenize_en, init_token='<sos>', unk_token='_UNK', eos_token='<eos>', lower=True,
                 batch_first=True, include_lengths=True)

def _get_textfield_snli():
    def _tokenize_en(text):
        return [x for x in text.split(" ") if x != "" and x.find(" ") == -1]
    return Field(tokenize=_tokenize_en, init_token='<sos>', unk_token='<unk>', eos_token='<eos>', lower=True,
                 batch_first=True, include_lengths=True)


def get_ptb(root):
    text_field = _get_textfield()
    trainset = TranslationDataset(root+'ptb/ptb.train', exts=('.txt', '.txt'), fields=(text_field, text_field))
    validset = TranslationDataset(root+'ptb/ptb.valid', exts=('.txt', '.txt'), fields=(text_field, text_field))
    testset = TranslationDataset(root+'ptb/ptb.test', exts=('.txt', '.txt'), fields=(text_field, text_field))
    text_field.build_vocab(trainset)
    return trainset, validset, testset

# def get_wiki103(root):
#     text_field = _get_textfield()
#     trainset = LanguageModelingDataset(root+'wikitext-103/wiki.train.tokens', text_field)
#     validset = LanguageModelingDataset(root+'wikitext-103/wiki.valid.tokens',  text_field)
#     testset = LanguageModelingDataset(root+'wikitext-103/wiki.test.tokens', text_field)
#     text_field.build_vocab(trainset)
#     return trainset, validset, testset

def get_yahoo(root):

    text_field = _get_textfield()
    trainset = TranslationDataset(root+'yahoo/yahoo.train', exts=('.txt', '.txt'), fields=(text_field, text_field))
    validset = TranslationDataset(root+'yahoo/yahoo.valid', exts=('.txt', '.txt'), fields=(text_field, text_field))
    testset = TranslationDataset(root+'yahoo/yahoo.test', exts=('.txt', '.txt'), fields=(text_field, text_field))
    text_field.build_vocab(trainset)
    return trainset, validset, testset

def get_yelp(root):

    text_field = _get_textfield()
    trainset = TranslationDataset(root+'yelp/yelp.train', exts=('.txt', '.txt'), fields=(text_field, text_field))
    validset = TranslationDataset(root+'yelp/yelp.valid', exts=('.txt', '.txt'), fields=(text_field, text_field))
    testset = TranslationDataset(root+'yelp/yelp.test', exts=('.txt', '.txt'), fields=(text_field, text_field))
    text_field.build_vocab(trainset)
    return trainset, validset, testset

def get_snli(root):

    text_field = _get_textfield_snli()
    trainset = TranslationDataset(root+'snli/snli.train', exts=('.txt', '.txt'), fields=(text_field, text_field))
    validset = TranslationDataset(root+'snli/snli.valid', exts=('.txt', '.txt'), fields=(text_field, text_field))
    testset = TranslationDataset(root+'snli/snli.test', exts=('.txt', '.txt'), fields=(text_field, text_field))
    text_field.build_vocab(trainset)
    return trainset, validset, testset


def get_dataset(root, dataset, seq_length=100, transform=_dummy_transform):
    assert dataset in ["BouncingBalls", "KTH", "PTB", "Yahoo", "Yelp", "SNLI"]
    trainset = validset = testset = None
    if dataset == "BouncingBalls":
        trainset, validset, testset = get_bouncing_balls(root, seq_length, transform)
    elif dataset == "KTH":
        trainset, validset, testset = get_kth(root, seq_length, transform)
    elif dataset == "PTB":
        trainset, validset, testset = get_ptb(root)
    elif dataset == "Yahoo":
        trainset, validset, testset = get_yahoo(root)
    elif dataset == "Yelp":
        trainset, validset, testset = get_yelp(root)
    elif dataset == "SNLI":
        trainset, validset, testset = get_snli(root)
    return trainset, validset, testset
