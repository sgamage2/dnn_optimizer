""" This optimizer keeps a separate learning rate for the set of all incoming weights to a neuron
The learning rates are stored in Tensors of shape of Parameters (weights and biases of each layer)
 """


import torch
from torch.optim import Optimizer
import numpy as np


# ToDo: optimize this function
def get_entropy(weights):
    n = len(weights)
    weights_arr = weights.cpu().numpy()
    hist = np.histogram(weights_arr, bins=30)  # getting the histogram
    # print(hist)
    hist_1 = [x / n for x in hist[0]]   # ToDo: Optimize

    entropy = 0
    for x in hist_1:       # ToDo: Optimize
        if x > 0:
            entropy -= x * np.log(np.abs(x))

    return entropy


class PerNeuronLR(Optimizer):
    def __init__(self, params, lr):
        assert lr > 0
        defaults = dict(lr=lr)
        super(PerNeuronLR, self).__init__(params, defaults)

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

                self.set_per_neuron_lr(p)   # Updates the learning rate tensors in self.state

                state = self.state[p]
                per_weight_lr = state['lr']
                assert p.shape == per_weight_lr.shape

                weight_change = -per_weight_lr * d_p    # Tensor of same shape as p and d_p

                # For inplace multiplication, we can do per_weight_lr.mul(-d_p)
                # But this will vary the stored learning rates through the optimization steps

                p.add_(weight_change)      # SGD update rule: p = p - lr * d_p

        loss = None
        return loss

    def set_per_neuron_lr(self, p):
        state = self.state[p]
        per_weight_lr = state['lr']
        assert p.shape == per_weight_lr.shape

        # For the connection between two layers L1 --> L2
        if p.ndim == 1:     # This Parameter is a vector of biases (shape = L2_nodes)
            pass    # We ignore bias params for now (initial lr is kept constant)
        elif p.ndim == 2:   # This Parameter is a Tensor of weights (shape = L2_nodes, L1_nodes)
            L2_nodes = p.shape[0]
            for row in range(L2_nodes):
                neuron_weights = p[row, :]
                # entropy = get_entropy(neuron_weights)
                entropy = np.random.randn()   # For now, get some random number
                per_weight_lr[row, :] = per_weight_lr[row, :] * entropy # Change learning rate based on entropy
