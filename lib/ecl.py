import torch
import torch.autograd as autograd


class EuclideanProjection(autograd.Function):

    @staticmethod
    def forward(ctx, x, k):
        # x = [bs, seq_len], k = [bs]
        bs, dim = x.shape
        assert (k.size(0) == bs) and (k.dim() == 1)
        eps = 1e-3
        gridsize = 100
        max_iter = 3
        iter_func = EuclideanProjection.get_iterfunc(eps, gridsize, x, k)
        xsorted = torch.sort(x, -1, descending=True).values
        nulow = (-1 * torch.gather(xsorted, -1, k.unsqueeze(1) - 1)).squeeze()
        nuup = nulow + 1.0

        for _ in range(max_iter):
            nulow, nuup = iter_func(nulow, nuup)
        # Given (approx) optimal nu, compute optimal y.
        nu = (nulow + nuup) / 2
        y = (x + nu.unsqueeze(1)).clip(min=0.0, max=1.0)

        ctx.save_for_backward(x, k, y)
        ctx.nu = nu
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, k, y = ctx.saved_tensors
        nu = ctx.nu
        grad_x = torch.zeros_like(x)
        mask = torch.logical_and((0 < y), (y < 1)) # selects fractional elements
        # Compute dv
        dv = (mask.type_as(x) * grad_output).sum(-1) / mask.sum(-1)
        grad_x = torch.where(mask, grad_output - dv.reshape(-1, 1), grad_x)
        return grad_x, None

    @staticmethod
    def get_iterfunc(eps, gridsize, x, k):
        x = x.detach() # detach should not be necessary as called within autograd.Function
        k = k.detach().type_as(x)
        bs, dim = x.shape
        dtype, device = x.dtype, x.device
        grid_01 = torch.linspace(0,1, gridsize, dtype=dtype, device=device)
        # [gridsize]
        zeros = x.new_zeros((bs, 1))

        def iter(nulow, nuup):
            # takes current lower and upper bound for nu, and brings them closer.
            delta = (nuup - nulow)
            mask = delta > eps  # only improve bound if sufficiently far away
            if mask.sum() == 0:
                # abort iteration, hacky.
                return nulow, nuup
            nugrid = (grid_01.view(1, -1) * delta[mask].view(-1, 1) + nulow[mask].view(-1, 1))
            # print(nugrid.shape), check is # sum(mask) x gridsize
            # [1, gridsize] * [sum(mask), 1] + [sum(mask), 1]
            # delta[mask] => returns elements from delta for positions where mask is True only.
            # Broadcasting rules are applied here but did not understand what is being achieved by
            # here.
            # Resulting tensor dimensions: for each dimension size, the resulting dimension
            # size is the max of the sizes of x and y along that dimension.
            # [sum(mask) * gridsize]
            _x = x[mask].unsqueeze(1) + nugrid.unsqueeze(-1)  # sum(mask) x gridsize x dim
            res = _x.clip(min=0, max=1).sum(-1)
            upix = torch.searchsorted(res, k[mask].unsqueeze(1))  # is rightmost entry
            upix = upix.clip(min=1, max=99)  # TODO: Added max=99 to avoid CUDA assertion
            nuup[mask] = torch.gather(nugrid[mask], -1, upix).squeeze()
            nulow[mask] = torch.gather(nugrid[mask], -1, upix-1).squeeze()

            return nulow, nuup

        return iter


class ECL(torch.nn.Module):

    def forward(self, input, k, use_soft_mask=False):
        # input = [bs, seq_len]
        # k = [bs]
        soft_output = EuclideanProjection.apply(input, k)
        if use_soft_mask:
            return soft_output
        _, selected = torch.topk(soft_output, k[0])
        hard_output = torch.zeros_like(soft_output)
        hard_output = hard_output.scatter_(-1, selected, 1)
        # print("k=",self.k, " | selected shape =", selected.shape)
        # print("hard_output=", hard_output)
        result = torch.autograd.Variable(hard_output - soft_output.data, requires_grad=True) + soft_output
        if torch.abs(result.sum() - hard_output.sum()) > 0.0001:
            print("Warning: instability in straight-through estimator!")
        return result


def test():
    eucl_proj = EuclideanProjection.apply
    bs = 5
    dim = 20
    x = torch.randn(bs, dim)
    k = torch.randint(low=1, high=dim, size=(bs,))
    x.requires_grad = True
    y = eucl_proj(x, k)
    # Some transformation.
    loss = torch.sigmoid(y.pow(2)).sum()
    loss.backward()
    print(x.grad)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    test()
