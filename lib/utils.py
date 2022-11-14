import torch
from torch import nn
import numpy as np
from scipy import signal
import math
import pytorch_lightning as pl
from torch.nn.utils import weight_norm
from torch.autograd import Function


# ------------------------------------ Video evaluation metrics ------------------------------------------------------ #
# ssim function is borrowed from Babaeizadeh et al. (2017), Fin et al. (2016)
# the implemenation is taken from: https://github.com/edenton/svg/blob/master/utils.py

def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                res = finn_ssim(gt[t][i][c], pred[t][i][c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr


def finn_psnr(x, y):
    mse = ((x - y) ** 2).mean()
    return 10 * np.log(1 / mse) / np.log(10)


def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def finn_ssim(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


# ------------------------------------ Neural network utilities ------------------------------------------------------ #

def weights_init(module):
    """ Weight initialization for different neural network components. """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.orthogonal_(module.weight)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        module.weight.data.normal_(0.0, 0.02)


def WN(module, norm=True):
    classname = module.__class__.__name__
    if norm:
        if classname.find('LSTMcell') != -1 or classname.find('GRUcell') != -1:
            module = weight_norm(module, dim=0, name='weight_ih')
            module = weight_norm(module, dim=0, name='weight_hh')
            return module
        else:
            return module
    else:
        return module


class Crop2d(nn.Module):

    def __init__(self, num):
        super().__init__()
        self.num = num

    def forward(self, input):
        if self.num == 0:
            return input
        else:
            return input[:, :, self.num:-self.num, self.num:-self.num]


class EMA(nn.Module):
    """ Exponential Moving Average.
    Note that we store shadow params as learnable parameters. We force torch.save() to store them properly..
    """

    def __init__(self, model, decay):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('EMA decay must in [0,1]')
        self.decay = decay
        self.shadow_params = nn.ParameterDict({})
        self.train_params = {}
        if decay == 0:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[self._dotless(name)] = nn.Parameter(param.data.clone(), requires_grad=False)

    @staticmethod
    def _dotless(name):
        return name.replace('.', '^')

    @torch.no_grad()
    def update(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[self._dotless(name)].data = \
                        self.decay * self.shadow_params[self._dotless(name)].data + (1.0 - self.decay) * param.data

    @torch.no_grad()
    def assign(self, model):
        # ema assignment
        train_params_has_items = bool(self.train_params)
        if self.decay > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if train_params_has_items:
                        self.train_params[name].data.copy_(param.data)
                    else:
                        self.train_params[name] = param.data.clone()
                    param.data.copy_(self.shadow_params[self._dotless(name)].data)

    @torch.no_grad()
    def restore(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.train_params[name].data)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RevGrad(nn.Module):
    def __init__(self, scaler=1.0):
        super().__init__()
        self.scaler = scaler

    def forward(self, input_):
        return RevGradFunction.apply(input_, self.scaler)


class RevGradFunction(Function):

    @staticmethod
    def forward(ctx, input, scaler=1.0):
        ctx.scaler = scaler
        return input

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output.neg() * ctx.scaler, None

# ------------------------------------ Other utilities --------------------------------------------------------------- #


def run_cuda_diagnostics(requested_num_gpus):
    print("CUDA available: ", torch.cuda.is_available())
    print("Requested num devices: ", requested_num_gpus)
    print("Available num of devices: ", torch.cuda.device_count())
    print("CUDNN backend: ", torch.backends.cudnn.enabled)
    assert requested_num_gpus <= torch.cuda.device_count(), "Not enough GPUs available."


def make_reproducible(seed):
    pl.seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)


def log_stdout(variable, text, frequency, step):
    if frequency > 0 and step % frequency == 0:
        print("[step=",step,"]")
        print(text, " = ", variable)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
