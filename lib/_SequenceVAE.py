from abc import ABC, abstractmethod
import pytorch_lightning as pl
from data.get_dataset import get_dataset
from lib.probability import IsoGaussian, IsoGaussianFixedSTD
from lib.utils import weights_init, EMA, RevGradFunction, log_stdout
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import LSTMCell
from torch.optim import *
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from lib.ecl import ECL


class _SequenceVAE(pl.LightningModule, ABC):

    def __init__(self, seed, lrate, lrate_decay, embed_dim, z_dim, hid_dim, batch_size, dataset, root,
                 obs_model, kl_rate, seq_length, tf_level, decoder_dropout, wdp, kl_start, hid_dropout, ema_rate,
                 free_bits, ddr, use_soft_mask, lambd, random_scores, **kw):
        super().__init__()

        #assert np.count_nonzero(np.array([wdp, ddr, decoder_dropout])) <= 1, "More than 1 dropout technique activated!"
        assert 0 <= wdp <= 1 and 0 <= ddr <= 1 and 0 <= decoder_dropout <= 1, "Invalid dropout value!"

        # save HPs to checkpoints
        self.training_outputs = []
        #self.automatic_optimization = False
        self.save_hyperparameters()
        self.dataset = dataset
        # signature
        self.signature_string = '_[rst-{}]-s-{}-lr-{}_lrd-{}_b-{}_h-{}_z-{}_klr-{}-tf-{}_dp-{}_wdp-{}_hdp-{}_ema-{}_fb-{}_ddr-{}_lmbd-{}_sm-{}_ddlr-{}_adam-{}' \
            .format(random_scores, seed, lrate, lrate_decay, batch_size, hid_dim, z_dim, kl_rate, tf_level,
                    decoder_dropout, wdp, hid_dropout, ema_rate, free_bits, ddr, lambd, use_soft_mask,
                    self.hparams.dd_lrate, self.hparams.use_adam)

        # load data and cut the sequences if required
        # seq_length = 100 default
        self.trainset, self.valset, self.testset = get_dataset(root=root, dataset=dataset, seq_length=seq_length,
                                                               transform=obs_model.normalize)

        # initialize variables that PERSIST in checkpoints
        self.register_buffer('lrate', torch.ones(1) * lrate)  # we make lrates persistent due to bugs in PL
        self.register_buffer('dd_lrate', torch.ones(1) * self.hparams.dd_lrate)
        self.register_buffer('kl_z_coeff', torch.ones(1)*kl_start)
        self.register_buffer('tf_level', torch.ones(1)*tf_level)

        # initialize probability models
        self.prior_model = self.post_model = IsoGaussian()
        self.score_model = IsoGaussianFixedSTD()
        self.obs_model = obs_model

        # transformer encoder
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=hid_dim,
        #     nhead=4,
        #     dim_feedforward=2048,
        #     batch_first=True
        # )

        # RNN encoder
        self.rnn_encoder = nn.LSTM(embed_dim, hid_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.hid2zparams = nn.Linear(hid_dim, z_dim * self.post_model.params_per_dim(), bias=False)

        # RNN decoder
        self.z2state = nn.Linear(z_dim, hid_dim, bias=False)
        self.rnn_predict_cell = LSTMCell(z_dim, hid_dim)
        self.rnn_update_cell = LSTMCell(embed_dim + z_dim, hid_dim)
        self.rnn_update_cell2 = LSTMCell(hid_dim, hid_dim)
        ## Cannot add number of layers to LSTMCell directly, hence adding another layer/sequence
        ## of cells manually. This layer will take in the output of the first LSTM upate cell.
        ## This is primarily to test out the role of cell state in encoding of the information.
        ## By adding this layer, we can manually add dropout to LSTM cell transitions both in the 
        ## first layer (kind of word level) and in the second layer (with hope that second layer
        ## captures sentence level features.) As it stands, do not use double lstm, so rnn_predict
        ## cell becomes redundant. 
        self.lstm_dropout1 = nn.Dropout(0.4)
        self.lstm_dropout2 = nn.Dropout(0.4)
        ## dropouts for hidden and cell states for both the layers.
        ## Possibly try with different dropout rates for hidden and cell too. Keeping same for now.

        self.decoder_dropout = nn.Dropout(decoder_dropout)

        # dynamic dropout
        self.dd_euclidean_projection = ECL()
        self.dd_lstm = nn.LSTM(embed_dim, hid_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.dd_score_predictor = nn.Linear(hid_dim, self.score_model.params_per_dim(), bias=False)


    # -----------------------------------------------------
    # Engineering and PyTorch Lightning-related methods
    # -----------------------------------------------------

    def configure_optimizers(self):
        params = list(self.named_parameters())
        def is_dd(n): return 'dd_' in n
        dd_params = [p for n, p in params if is_dd(n)]  # parameters of dynamic dropout layers
        vae_params = [p for n, p in params if not is_dd(n)]  # parameters of VAE layers
        if self.hparams.use_adam:
            self.vae_optimizer = SGD(vae_params, lr=self.lrate.item())
            self.dd_optimizer = Adam(dd_params, lr=self.dd_lrate.item())
        else:
            self.vae_optimizer = SGD([
                {'params': vae_params},
                {'params': dd_params, 'lr': self.dd_lrate.item()}
            ], lr=self.lrate.item())
        self.vae_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.vae_optimizer, gamma=self.hparams.lrate_decay)
        self.configure_ema()
        return [self.vae_optimizer], [self.vae_scheduler]

    def training_step(self, batch, batch_idx):
        self.apply_annealing()
        self.vae_optimizer.zero_grad()
        self.dd_optimizer.zero_grad() if self.hparams.use_adam else None
        sequence, sequence_len = self._get_sequence_from_batch(batch)
        #print("Sequence")
        #print(sequence.shape)
        #print(sequence_len.shape)
        #print(sequence[:3])
        #print(sequence_len[:10])
        output = self(sequence, sequence_len)
        self.manual_backward(output['loss'].mean(), self.vae_optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.vae_optimizer.step()
        self.dd_optimizer.step() if self.hparams.use_adam else None
        self.ema.update(self)
        self.training_outputs.append({'loss': output['loss'].mean().detach().cpu(),
                                      'logpxz': output['logpxz'].mean().detach().cpu(),
                                      'kl_z': output['kl_z'].mean().detach().cpu(),
                                      'kl_scores': output['kl_scores'].mean().detach().cpu()})

    def training_epoch_end(self, outputs):
        outputs = self.training_outputs
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        logpxz = torch.stack([x['logpxz'] for x in outputs]).mean()
        kl_z = torch.stack([x['kl_z'] for x in outputs]).mean()
        kl_scores = torch.stack([x['kl_scores'] for x in outputs]).mean()
        self.logger.log_metrics({'Iteration': self.global_step,
                                 'Training [total]': loss,
                                 'Training [-elbo]': -logpxz + kl_z,
                                 'Training [-logpxz]': -logpxz,
                                 'Training [kl_z]': kl_z,
                                 'Training [kl_scores]': kl_scores,
                                 'KL(z) Coeff': self.kl_z_coeff.item(),
                                 'TF level': self.tf_level.item()})
        self.vae_scheduler.step()
        if self.hparams.interrupt_step < self.global_step:
            print("Early stopping")
            exit()
        self.training_outputs = []

    def validation_step(self, batch, batch_idx):
        self.ema_assign() if batch_idx == 0 else None
        sequence, sequence_len = self._get_sequence_from_batch(batch)
        output = self(sequence, sequence_len)
        return {'loss': output['loss'], 'logpxz': output['logpxz'], 'kl_z': output['kl_z']}

    def validation_epoch_end(self, outputs):
        print("Output size: ", len(outputs))
        print(len(self.testset)/32)
        loss = torch.cat([x['loss'] for x in outputs]).mean()
        logpxz = torch.cat([x['logpxz'] for x in outputs]).mean()
        kl_z = torch.cat([x['kl_z'] for x in outputs]).mean()
        total_kl = torch.cat([x['kl_z'] for x in outputs]).sum()
        total_rec = torch.cat([x['logpxz'] for x in outputs]).sum()
        mi = self.estimate_mutual_information()  # removed not to slow down evaluation
        self.ema_restore()
        lr = 0
        #print(self.optimizers())
        for pg in self.optimizers().param_groups:
            lr = pg['lr']
        self.logger.log_metrics({'Iteration': self.global_step,
                                 'Validation [total]': loss,
                                 'Validation [-elbo]': -logpxz + kl_z,
                                 'Validation [-logpxz]': -logpxz,
                                 'Validation [kl_z]': kl_z,  # 'Validation [MI]': mi,
                                 'Learning rate': lr})  # Here we are logging the learning rate because PL is not...
        
        self.log('neg_elbo', -logpxz + kl_z)
        if self.dataset == "PTB":
            print("PPL: ", torch.exp((-total_rec + total_kl)/82430))
        elif self.dataset == "SNLI":
            print("PPL: ", torch.exp((-total_rec + total_kl)/106795))
        elif self.dataset == "Yahoo":
            print("PPL: ", torch.exp((-total_rec + total_kl)/798673))
        print("NELBO: ", -logpxz + kl_z)
        print("Rec/NLL: ", -logpxz)
        print("KL: ", kl_z)
        print("MI: ", mi)

    def configure_ema(self):
        self.ema = EMA(self, self.hparams.ema_rate)

    def ema_assign(self):
        self.ema.assign(self)

    def ema_restore(self):
        self.ema.restore(self)

    def train_dataloader(self):
        return DataLoader(self.trainset, self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        dataset = self.testset if self.hparams.use_testset else self.valset
        return DataLoader(dataset, self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.testset, self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def apply_annealing(self):
        if self.training:
            self.kl_z_coeff = min(torch.ones(1, device=self.device), self.kl_z_coeff + self.hparams.kl_rate)

    def init_weights(self):
        self.apply(weights_init)

    # -----------------------------------------------------
    # VAE logic
    # -----------------------------------------------------

    def forward(self, sequence, sequence_len=None):
        """ Sequence format: [batch_size, num_steps, dim1, dim2, ...]. """
        # sequence: [bs, seq_len]
        # initialize stuff
        logpxz = torch.zeros(sequence.size(0), device=self.device)
        reconstruction = []
        x_params_list = []
        z_params, _ = self.encode(sequence, sequence_len)
        # Applies word embedding => LSTM layer => takes the hidden part of the lstm output and passes
        # it to a linear layer with out_features = 2*z_dim.
        # [bs, z_dim*2], [bs, seq_len, hid_dim]
        # encode sequence with 'reparameterization trick'
        self.z, kl_z = self.post_model.reparameterize(params=z_params, free_bits=self.hparams.free_bits,
                                                      test_sampling=True)
        # [bs, z_dim], [bs]

        # dynamic dropout
        if self.hparams.ddr > 0 and self.training:
            mask, kl_scores = self.compute_mask(sequence, sequence_len)
            # mask = [32, 41] 41 seems to be the sequence length
            # [bs, seq_len]
            # kl_scores = [32]
            #print("seq-len", sequence.size(1))
            #print("mask-shape",mask.shape)
            #print(kl_scores.shape)
        else:
            mask = None
            kl_scores = torch.zeros_like(logpxz)
        # infer initial generator-cell hidden-state from z
        state = self.z2state(self.z)
        # [bs, hid_dim]
        if isinstance(self.rnn_update_cell, LSTMCell):
            state = (state, torch.zeros_like(state))
            # ([bs, hid_dim], [bs, hid_dim])
            ## initializing the cell state to be zero
            ## TODO: Should the cell state be initialized same as the hidden state?
        # decode sequence observation-by-observation
        for i in range(sequence.size(1)): # iterate through time_steps dim
            x_params = self.decode_observation(state)
            # [bs, output_dim]
            x_params_list.append(x_params)
            prediction = self.obs_model.get_most_probable_output(x_params)
            # [bs], argmax across last dim
            reconstruction.append(prediction)
            input = self._get_next_input(truth=sequence[:, i], prediction=prediction)
            # [bs]
            state = self.decode_state(state=state, input=input, z=self.z, prediction=prediction,
                                      mask=mask[:, i].unsqueeze(-1) if mask is not None else None)
            # ([bs, hid_dim], [bs, hid_dim])
            logpxz += self.obs_model.conditional_log_prob(x_params, sequence[:, i])

        output = dict()
        output['kl_z'] = kl_z
        output['logpxz'] = logpxz
        output['state'] = state
        output['z'] = self.z
        output['z_mean'] = self.post_model.get_mean(z_params)
        output['z_logvar'] = self.post_model.get_logvar(z_params)
        output['kl_scores'] = kl_scores
        output['loss'] = -(logpxz - self.kl_z_coeff * kl_z) + self.hparams.lambd * kl_scores
        output['reconstruction'] = torch.stack(reconstruction, dim=1)
        output['output_params'] = torch.stack(x_params_list, dim=1)
        return output

    def decode_state(self, state, input, z, prediction=None, mask=None):
        # state = ([bs, hid_dim], [bs, hid_dim])
        # input = [bs]
        # mask = [bs, seq_len]
        # z = [bs, z_dim]
        # prediction = [bs]
        input = self.embed_input(input)
        # [bs, emb_dim]
        input = self.decoder_dropout(input)
        ## input dropout; nothing to do with cell states
        # predict part
        apriori_state = self.rnn_predict_cell(z, state) if (not self.hparams.do_not_use_double_lstm) else state
        # ([bs, hid_dim], [bs, hid_dim])
        ## state input here is: hidden vector calculated by applying a linear layer
        ## to the latent vector and the cell vector is a vector of zeros.
        ## Probably need to try the cell state experiments without double lstm first.
        # dynamic dropout
        if self.hparams.ddr > 0 and self.training and mask is not None:
            input = input * mask
        # update part
        # aposteriori_state = self.rnn_update_cell(torch.cat([input, z], dim=-1), apriori_state)
        hidden1, cell1 = self.rnn_update_cell(torch.cat([input, z], dim=-1), apriori_state)
        # ([bs, hid_dim], [bs, hid_dim])
        ## apply dropout to hidden and cell state here
        hidden1, cell1 = self.lstm_dropout1(hidden1), self.lstm_dropout1(cell1)

        hidden2, cell2 = self.rnn_update_cell2(hidden1, (hidden1, cell1))
        ## apply dropout to hidden and cell state in the 2nd layer here

        hidden2, cell2 = self.lstm_dropout2(hidden2), self.lstm_dropout2(cell2)


        
        # return new state
        return (hidden2, cell2)

    @abstractmethod
    def _get_sequence_from_batch(self, batch):
        """ Embed input before forwarding it to a recurrent cell. """

    @abstractmethod
    def embed_input(self, input):
        """ Embed input before forwarding it to a recurrent cell. """

    @abstractmethod
    def decode_observation(self, state):
        """ Predict observation. """

    def encode(self, sequence, sequence_len=None):
        """ Encode parameters of the posterior.  Sequence format: [batch_size, num_steps, dim1, dim2, ...]. """
        # sequence = [bs, seq_len], sequence_len = [bs]
        embedded_sequence = self.embed_input(sequence)
        # [bs, seq_len, emb_dim]

        if sequence_len is not None:
            embedded_sequence = pack_padded_sequence(embedded_sequence, sequence_len.cpu(), batch_first=True)
        output, hidden = self.rnn_encoder(embedded_sequence)
        # The output is a PackedSequence object from which we need to separate the output and
        # hidden tensors. If packing had not been used, the dimensions would be as follows:
        # output = [bs, seq_len, hid_dim]
        # hidden = ([1, bs, hid_dim], [1, bs, hid_dim])
        if isinstance(self.rnn_encoder, nn.LSTM):
            hidden = hidden[0]
            # [1, bs, hid_dim]
            # hidden is not affected by packing, so it can be taken out directly

            ## only taking out the hidden state over here. If you need to manipulate the cell state
            ## later on, you might need to come back here.
            ## This is encoder side. Nothing to do with dropout here.
        if sequence_len is not None:
           output, _ = pad_packed_sequence(output, batch_first=True)
           # Getting the original format tensors again from packing.
           # [bs, seq_len, hid_dim]
        return self.hid2zparams(hidden.squeeze()), output
        # [bs, z_dim*2], [bs, seq_len, hid_dim]

    def _get_next_input(self, truth, prediction):
        # input = [bs], prediction = [bs]
        input = prediction if (random.uniform(0, 1) > self.tf_level and self.training) else truth
        if random.uniform(0, 1) < self.hparams.wdp and self.training:
            input = torch.zeros_like(input)
        # doubt: what's the difference between tf_level and wdp
        return input

    # -----------------------------------------------------
    # Dynamic dropout logic
    # -----------------------------------------------------

    def _infer_dd_context(self, sequence, sequence_len):
        embedded_sequence = self._dd_embed_input(sequence)
        # [bs, seq_len, emb_dim]
        if sequence_len is not None:
            embedded_sequence = pack_padded_sequence(embedded_sequence, sequence_len.cpu(), batch_first=True)
            # PackedSquence object with data and batch sizes as attrs
        enc_output, _ = self.dd_lstm(embedded_sequence)
        if sequence_len is not None:
            enc_output, _ = pad_packed_sequence(enc_output, batch_first=True)
            # [bs, seq_len, hid_dim]
        #log_stdout(enc_output.shape, "enc output shape", self.hparams.verbose_frequency, self.global_step)
        return enc_output

    def compute_mask(self, sequence, sequence_len):
        # sequence = [bs, seq_len]
        # sequence_len = [bs], lengths of all sequences
        scores, kl_scores = self.compute_scores(sequence, sequence_len)
        # [bs, seq_len], [bs]
        use_soft_mask = False if not self.training else self.hparams.use_soft_mask
        k_vector = torch.round(sequence_len * (1 - self.hparams.ddr)).to(torch.int64)
        # rounded to the closest integer
        # sequence_len = [bs]
        #log_stdout(k_vector, "k-vector", self.hparams.verbose_frequency, self.global_step)
        mask = self.dd_euclidean_projection(scores, k_vector, use_soft_mask)
        #log_stdout(mask[0], "mask", self.hparams.verbose_frequency, self.global_step)
        return RevGradFunction.apply(mask), kl_scores

    def compute_scores(self, sequence, sequence_len):
        context = self._infer_dd_context(sequence, sequence_len)  # [batch_size x sequence_size x 2*embed_dim]
        # [bs, seq_len, hid_dim]
        score_params = self.dd_score_predictor(context)  # [batch_size x sequence_size x 2]
        # [bs, seq_len, 1]
        #log_stdout(score_params[0], "mean scores", self.hparams.verbose_frequency, self.global_step)
        scores, kl_scores = self.score_model.reparameterize(score_params, test_sampling=False)
        # scores = [bs, seq_len]
        # kl_scores = [bs]
        if self.hparams.random_scores:  # for debugging purposes
            scores = torch.rand_like(scores)
            kl_scores = torch.zeros_like(kl_scores)
        #log_stdout(scores[0], "scores", self.hparams.verbose_frequency, self.global_step)
        return scores, kl_scores

    @abstractmethod
    def _dd_embed_input(self, input):
        """ Embed input before forwarding it to dynamic dropout. """

    # -----------------------------------------------------
    # Evaluation-time methods
    # -----------------------------------------------------

    @torch.no_grad()
    def signature(self):
        return self.signature_string

    @torch.no_grad()
    def sample(self, steps, z=None):
        """ Sample a random sequence. """
        # sample random z if not given already
        if z is None:
            z = self.prior_model.sample_from_prior([1, self.hparams.z_dim]).to(self.device)
        # infer initial generator-cell hidden-state from z
        state = self.z2state(z)
        if isinstance(self.rnn_update_cell, LSTMCell):
            state = (state, torch.zeros_like(state))
        # predict specified number of steps
        pred_list = []
        for i in range(steps):
            pred_params = self.decode_observation(state)
            pred_list.append(self.obs_model.get_most_probable_output(pred_params))
            state = self.decode_state(state=state, input=pred_list[-1], z=z, prediction=pred_list[-1])
        return torch.stack(pred_list, dim=1)

    @torch.no_grad()
    def get_latent_representation(self, sequence, sequence_len=None):
        z_params, _ = self.encode(sequence.to(self.device), sequence_len.to(self.device))
        return self.post_model.get_mean(z_params)

    @torch.no_grad()
    def estimate_mutual_information(self):
        # code adapted from: https://github.com/yoonkim/skip-vae
        all_z = []
        all_kl = []
        means = []
        logvars = []
        for batch_idx, batch in enumerate(self.test_dataloader()):
            sequence, sequence_len = self._get_sequence_from_batch(batch)
            sequence = sequence.to(self.device)
            sequence_len = sequence_len.to(self.device)
            output = self(sequence, sequence_len)
            all_kl.append(output['kl_z'])
            all_z.append(output['z'])
            means.append(output['z_mean'])
            logvars.append(output['z_logvar'])
        mean_prior = torch.zeros(1, self.hparams.z_dim, device=self.device)
        logvar_prior = torch.zeros(1, self.hparams.z_dim, device=self.device)
        all_z = torch.cat(all_z)
        all_kl = torch.cat(all_kl)
        means = torch.cat(means)
        logvars = torch.cat(logvars)
        agg_kl = torch.zeros(1, device=self.device)
        def log_gaussian(sample, mean, logvar):
            dist = torch.distributions.Normal(mean, torch.exp(0.5 * logvar))
            return dist.log_prob(sample)
        N = all_z.size(0)
        for i in range(N):
            z_i = all_z[i].unsqueeze(0).expand_as(all_z)
            log_agg_density = log_gaussian(z_i, means, logvars)  # log q(z|x) for all x
            log_q = torch.logsumexp(log_agg_density, dim=0) - np.log(N*1.0)
            log_p = log_gaussian(all_z[i].unsqueeze(0), mean_prior, logvar_prior)
            agg_kl += log_q.sum() - log_p.sum()
        if agg_kl < 0:
            print("KL(agg) < 0....clamping to 0. This is due to numerical instability.")
            agg_kl = torch.clamp(agg_kl, min=0)
        mi = all_kl.mean() - agg_kl * 1.0 / N
        return mi.item()
