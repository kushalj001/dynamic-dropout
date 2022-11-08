from lib._SequenceVAE import _SequenceVAE
from torchtext.data import BucketIterator
import torch
from torch import nn
from lib.probability import Categorical


class SentenceVAE(_SequenceVAE):
    def __init__(self, embed_dim=512, obs_model=Categorical(), **kw):
        super().__init__(embed_dim=embed_dim, obs_model=obs_model, **kw)

        # initialization
        text_field = self.trainset.fields['src']
        self.vocab = text_field.vocab
        input_dim = output_dim = len(text_field.vocab)
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]
        self.init_idx = text_field.vocab.stoi[text_field.init_token]
        self.unk_idx = text_field.vocab.stoi[text_field.unk_token]
        self.eos_idx = text_field.vocab.stoi[text_field.eos_token]
        self.word_decoder = nn.Linear(self.hparams.hid_dim, output_dim, bias=False)
        self.word_encoder = nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim, padding_idx=self.pad_idx)
        self.dd_word_encoder = nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim, padding_idx=self.pad_idx)
        self.obs_model = Categorical(pad_idx=self.pad_idx, init_idx=self.init_idx, eos_idx=self.eos_idx)
        self.hid_dp = nn.Dropout(self.hparams.hid_dropout)

    def embed_input(self, input):
        return self.word_encoder(input)

    def _dd_embed_input(self, input):
        return self.dd_word_encoder(input)

    def decode_observation(self, state):
        # state = ([bs, hid_dim], [bs, hid_dim])
        state = state[0] if type(state) == tuple else state
        # state = self.hid_dp(state)
        ## applying dropout in the decode_state method on hidden and cell state already.
        ## don't want double dropout
        return self.word_decoder(state)
        # [bs, output_dim]

    def train_dataloader(self):
        return BucketIterator(self.trainset,  batch_size=self.hparams.batch_size, sort_within_batch=True)

    def val_dataloader(self):
        dataset = self.testset if self.hparams.use_testset else self.valset
        return BucketIterator(dataset, batch_size=self.hparams.batch_size, sort_within_batch=True)

    def test_dataloader(self):
        return BucketIterator(self.testset,  batch_size=self.hparams.batch_size, sort_within_batch=True)

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.01, 0.01)
            else:
                nn.init.constant_(param.data, 0)
        nn.init.uniform_(self.word_encoder.weight, -0.1, 0.1)

    def _get_next_input(self, truth, prediction):
        truth = truth.clone()
        truth[truth == self.eos_idx] = self.pad_idx
        return super()._get_next_input(truth, prediction)

    def _get_sequence_from_batch(self, batch):
        sequence, sequence_len = batch.src
        return sequence, sequence_len

    def validation_epoch_end(self, outputs):
        """ print 10 random sentences in addition to regular logging. """
        super().validation_epoch_end(outputs)
        for i in range(10):
            print("Randomly generated sentence", i+1, ":", self.sample(steps=50))
            print("\n")
            self.saliency_analysis(element=i)

    @torch.no_grad()
    def sample(self, steps, z=None):
        """ samples a sequence and converts into text format.   """
        sequence = super().sample(steps=steps, z=z)
        sequence = list(sequence.squeeze(0).cpu().numpy())
        return self.sequence_to_sentence(sequence)

    @torch.no_grad()
    def sequence_to_sentence(self, sequence):
        sentence = ""
        for ind in sequence:
            if ind == self.eos_idx:
                break
            sentence = sentence + ' ' + self.vocab.itos[ind]
        return sentence

    @torch.no_grad()
    def saliency_analysis(self, element=0):
        if self.hparams.ddr > 0:
            # get batch with a single randomly chosen sentence from the test set
            batch = next(iter(self.train_dataloader()))
            sequence, sequence_len = batch.src
            sequence = sequence.to(self.device)
            sequence_len = sequence_len.to(self.device)
            sequence = sequence[element:element+2]
            sequence_len = sequence_len[element:element+2]
            mask, _ = self.compute_mask(sequence, sequence_len)
            mask = mask[0].squeeze()
            # print the sentence with the corresponding per-word saliency estimates
            print("mask=", mask)
            sentence = ""
            for e, word in enumerate(list(sequence[0].squeeze(0).cpu().numpy())):
                if word == self.eos_idx:
                    break
                sentence = sentence + ' ' + self.vocab.itos[word] + ' [' + str(torch.round(mask[e]).item()) + ']'
            print("Saliency analysis:", sentence)