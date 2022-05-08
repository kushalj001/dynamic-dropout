from abc import abstractmethod
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


class ProbabilisticModel(nn.Module):

    def __init__(self, **kw):
        super().__init__()

    @abstractmethod
    def params_per_dim(self):
        """ how many parameters are computed per dimension. """

    @abstractmethod
    def get_mean(self, params):
        """ get mean value for specified parameters. """

    def get_mode(self, params):
        """ mean by default. """
        return self.get_mean(params)

    def get_most_probable_output(self, params):
        """ mode by default. """
        return self.get_mode(params)

    @abstractmethod
    def sample_once(self, params, sampling_temperature=1.0):
        """ sample from the model given parameters. """

    def sample(self, params, num_times):
        """ sample multiple times. """
        samples = []
        for i in range(num_times):
            samples.append(self.sample_once(params))
        return samples

    def reparameterize(self, params):
        """ compute distance from prior and return sample """
        return self.sample_once(params), torch.tensor([0.0], device=self.device)

    def conditional_log_prob(self, params, sample):
        """ compute conditional nll for a given sample. """
        return torch.tensor([0.0], device=self.device)


class IsoGaussian(ProbabilisticModel):

    def __init__(self):
        super().__init__()

    @staticmethod
    def params_per_dim():
        """ we have mean and log-variance, so 2 parameters. """
        return 2

    @staticmethod
    def get_mean(params):
        """ Means are estimated parameters. """
        mean, _ = params.chunk(2, dim=-1)
        return mean

    @staticmethod
    def get_logvar(params):
        """ Log-variance. """
        _, lv = params.chunk(2, dim=-1)
        return lv

    def sample_once(self, params, sampling_temperature=1.0):
        mean, lv = params.chunk(2, dim=-1)
        dst = Normal(mean, torch.exp(0.5 * lv) * sampling_temperature)
        return dst.sample(torch.Size([1])).squeeze(0)

    def sample_from_prior(self, shape, sampling_temperature=1.0):
        """ Sample from prior. """
        mean, lv = torch.zeros(shape), torch.zeros(shape)
        std = torch.exp(0.5 * lv) * sampling_temperature
        eps = torch.randn_like(std)
        return mean + eps * std

    def reparameterize(self, params, reduce=True, free_bits=0, test_sampling=True):
        """ compute distance from prior and return sample """
        # params = [bs, z_dim*2]
        mean, lv = params.chunk(2, dim=-1)
        # mean = [bs, z_dim]
        # lv = [bs, z_dim], we assume the model outputs log variance of the normal dst by default
        dst = Normal(mean, torch.exp(0.5 * lv))
        # converting log variance into std
        # creating a normal distribution with mean and std as that output
        # by the neural network
        kl_div = kl_divergence(dst, Normal(0, 1))
        # [bs, z_dim], the second term in the ELBO loss
        kl_div = kl_div.view(kl_div.size(0), -1).sum(dim=1) if reduce else kl_div
        # [bs], summing up kl_divergence across all the dimensions
        kl_div = torch.clamp(kl_div, min=free_bits) if self.training else kl_div
        sample = dst.rsample() if (self.training or test_sampling) else mean
        # [bs, z_dim]
        return sample.squeeze(-1), kl_div
        # [bs, z_dim], [bs] (squeeze not making sense with assumed dimensions)


class IsoGaussianFixedSTD(ProbabilisticModel):

    def __init__(self):
        super().__init__()

    @staticmethod
    def params_per_dim(std=1.0):
        """ we mean only. """
        return 1

    @staticmethod
    def get_mean(params):
        return params

    def sample_once(self, params, sampling_temperature=1.0):
        mean = self.get_mean(params)
        dst = Normal(mean, torch.ones(1, device=params.device) * sampling_temperature)
        return dst.sample(torch.Size([1])).squeeze(0)

    def sample_from_prior(self, shape, sampling_temperature=1.0):
        """ Sample from prior. """
        mean = torch.zeros(shape)
        std = torch.ones(1, device=shape.device) * sampling_temperature
        eps = torch.randn_like(std)
        return mean + eps * std

    def reparameterize(self, params, reduce=True, free_bits=0, test_sampling=True):
        """ compute distance from prior and return sample """
        # params = [bs, seq_len, 1]
        mean = self.get_mean(params)
        # [bs, seq_len, 1]
        dst = Normal(mean, torch.ones(1, device=params.device))
        # The std of the normal distribution is not taken from the NN.
        # It is fixed to one. The mean however is taken from the NN.
        # This corresponds to sampling "important" words to score and mask.
        kl_div = kl_divergence(dst, Normal(0, 1))
        # [bs, seq_len, 1]
        kl_div = kl_div.view(kl_div.size(0), -1).sum(dim=1) if reduce else kl_div
        # [bs]
        kl_div = torch.clamp(kl_div, min=free_bits) if self.training else kl_div
        sample = dst.rsample() if (self.training or test_sampling) else mean
        # [bs, seq_len, 1]
        return sample.squeeze(-1), kl_div
        # [bs, seq_len], [bs]


class Categorical(ProbabilisticModel):
    """ Used for language modeling. """

    def __init__(self, pad_idx=-100, init_idx=-100, eos_idx=-100, **kw):
        super().__init__(**kw)
        self.pad_idx = pad_idx
        self.init_idx = init_idx
        self.eos_idx = eos_idx
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_idx)

    def get_mean(self, params):
        raise NotImplementedError

    def sample_once(self, params, sampling_temperature=1.0):
        raise NotImplementedError

    def params_per_dim(self):
        return 1

    def get_most_probable_output(self, params):
        return params.argmax(-1)

    def conditional_log_prob(self, params, sample):
        clp = -self.cross_entropy(params, sample)
        mask = torch.ones_like(clp)
        mask[sample == self.pad_idx] = 0  # do not compute loss on padding
        # NOTE: does this ever execute? Do you need to mask padded tokens explicitly?
        # Are there padded tokens at all in the examples?
        if not self.training:  # do not compute loss on other tokens at test time
            mask[sample == self.eos_idx] = 0
            mask[sample == self.init_idx] = 0
        return clp * mask

    @staticmethod
    def normalize(x):
        return x

    @staticmethod
    def unnormalize(x):
        return x
