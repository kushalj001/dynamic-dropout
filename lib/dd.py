from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GumbelSoftMax:
    # partially based on: https://github.com/ethanfetaya/NRI/blob/master/utils.py

    def __init__(self, prior_keep_rate: float = 0.5, hard: bool = True, eps: float = 1e-10):
        assert (0 <= prior_keep_rate <= 1), "Prior keep rate must be in [0,1]"
        self.tau = 1.0
        self.tau_min = 0.5
        self.tau_annealing_rate = 0.0001
        self.prior_keep_rate = prior_keep_rate
        self.hard = hard
        self.eps = eps

    def _kl_divergence(self, probs):
        prior_probs = torch.ones_like(probs) - self.prior_keep_rate
        prior_probs[..., 0] = self.prior_keep_rate
        log_prior = torch.log(prior_probs + self.eps)
        return probs * (torch.log(probs + self.eps) - log_prior)

    def _sample_gumbel(self, shape):
        U = torch.rand(shape).float()
        return - torch.log(self.eps - torch.log(U + self.eps))

    def _gumbel_softmax_sample(self, logits):
        gumbel_noise = self._sample_gumbel(logits.size()).to(logits.device)
        y = logits + Variable(gumbel_noise)
        return F.softmax(y / self.tau, dim=-1)

    def _gumbel_softmax(self, logits):
        y_soft = self._gumbel_softmax_sample(logits)
        if self.hard:
            shape = logits.size()
            _, k = y_soft.data.max(-1)
            y_hard = torch.zeros(*shape, device=y_soft.device)
            y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
            y = Variable(y_hard - y_soft.data, requires_grad=True) + y_soft
        else:
            y = y_soft
        return y

    def reparameterize(self, logits):
        self.tau = max(self.tau_min, self.tau - self.tau_annealing_rate)
        sample = self._gumbel_softmax(logits)
        probs = F.softmax(logits, dim=-1)
        kl_div = self._kl_divergence(probs)
        return sample, kl_div.view(kl_div.size(0), -1).sum(dim=1)


class DynamicWordDropout(nn.Module):

    def __init__(self, context_dim: int, beta: float = 1.0, prior_keep_rate: float = 0.1, mlp_hid_dim: int = 512, **kw):
        super().__init__()
        self.prob_model = GumbelSoftMax(prior_keep_rate=prior_keep_rate)
        if mlp_hid_dim > 0:
            self.infer_logits = nn.Sequential(nn.Linear(context_dim, mlp_hid_dim),
                                              nn.ReLU(True),
                                              nn.Linear(mlp_hid_dim, 2))
        else:
            self.infer_logits = nn.Sequential(nn.Linear(context_dim, 2))
        self.beta = beta
        self.keep_rate = 0
        self.kl_div = torch.zeros(1)

    def kl(self) -> Tensor:
        return self.beta * self.kl_div

    @property
    def kr(self) -> float:
        return self.keep_rate

    def forward(self, input: Tensor, context: Tensor) -> Tensor:
        return self.forward_training(input, context) if self.training else self.forward_inference(input)

    def forward_training(self, input: Tensor, context: Tensor) -> Tensor:
        logits = self.infer_logits(context)
        mask, self.kl_div = self.prob_model.reparameterize(logits)
        mask = mask[..., 0:1]
        self.keep_rate = mask.sum().item() / mask.nelement()
        return mask * input

    def forward_inference(self, input: Tensor) -> Tensor:
        self.kl_div = torch.zeros(input.size(0), device=input.device)
        return input

    def get_probs(self, context: Tensor) -> Tensor:
        logits = self.infer_logits(context)
        probs = F.softmax(logits, dim=-1)
        return probs
