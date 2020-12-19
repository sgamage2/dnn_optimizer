""" A simple implementation of the Stochastic Gradient Descent (SGD): without weight decay, momentum etc
Follows the PyTorch SGD source code
https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
 """


import torch
from torch.optim import Optimizer


class SimpleSGD(Optimizer):
    def __init__(self, params, lr):
        assert lr > 0
        defaults = dict(lr=lr)
        super(SimpleSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        # By default there is only one group. It has all the params of the network
        # The params are in Parameter objects. Each layer has 2 Parameter objects (biases and weights)
        # Parameter is a subclass of Tensor (with some convenient properties to make it params of a Module)
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:   # p = params. A Parameter object with biases or weights of a layer
                if p.grad is None:
                    continue
                d_p = p.grad
                p.add_(d_p, alpha=-lr)      # SGD update rule: p = p - lr * d_p

        loss = None
        return loss
