""" This optimizer keeps Tensors of shape of Parameters for the learning rate
We do not adapt the learning rate or do anything fancy with it. So it is equivalent to SGD
This class is just an example/ precursor to go into per-weight learning rates
ToDo: Delete when this class is no longer needed
 """


import torch
from torch.optim import Optimizer


class PerWeightLR(Optimizer):
    def __init__(self, params, lr):
        assert lr > 0
        defaults = dict(lr=lr)
        super(PerWeightLR, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]   # A dict that keeps state of each Parameter obj (biases and weights of layers)
                state['step'] = 0
                state['lr'] = torch.full_like(p, lr)    # A Tensor of same shape as Parameter filled with initial lr

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                state = self.state[p]
                per_weight_lr = state['lr']
                weight_change = -per_weight_lr * d_p    # Tensor of same shape as p and d_p

                # For inplace multiplication, we can do per_weight_lr.mul(-d_p)
                # But this will vary the stored learning rates through the optimization steps

                p.add_(weight_change)      # SGD update rule: p = p - lr * d_p

        loss = None
        return loss
