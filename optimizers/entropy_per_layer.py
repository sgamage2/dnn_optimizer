""" A simple implementation of the Stochastic Gradient Descent (SGD): without weight decay, momentum etc
Follows the PyTorch SGD source code
https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
 """

import numpy as np
import torch
from torch.optim import Optimizer
from utility import entropy


class EntropyPerLayer(Optimizer):
    def __init__(self, params, lr):
        assert lr > 0
        defaults = dict(lr=lr)
        super(EntropyPerLayer, self).__init__(params, defaults)

        for group in self.param_groups:
            group['entropy_hist'] = []

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        # The params are in Parameter objects. Each layer has 2 Parameter objects (biases and weights)
        for group in self.param_groups: # Each group is a layer
            lr = group['lr']
            for p in group['params']:   # p = params. A Parameter object with biases or weights of a layer
                if p.grad is None:
                    continue

                d_p = p.grad
                p.add_(d_p, alpha=-lr)      # SGD update rule: p = p - lr * d_p


        loss = None
        return loss

    def on_epoch_start(self):
        """Performs things such as updating learning rate based on entropy.
        This method should be called in the training loop at the end of every epoch
        """
        # The params are in Parameter objects. Each layer has 2 Parameter objects (biases and weights)
        for group in self.param_groups:  # Each group is a layer
            lr = group['lr']
            entropy_hist = group['entropy_hist']

            params_list = []
            for p in group['params']:  # p = params. A Parameter object with biases or weights of a layer
                # Convert Parameter to numpy array (because entropies are computed with numpy operations)
                np_arr = p.cpu().detach().numpy()
                if len(np_arr.shape) == 1:
                    np_arr = np_arr.reshape(-1, 1)
                params_list.append(np_arr)

            # params_list has the bias vector (L x 1) and weights matrix (L x L_prev) --> concatenate cols
            # if len(params_list) > 0:
            weights = np.concatenate(params_list, axis=1)
            ent = entropy.get_entropy(weights)

            # print(ent)
            # print(lr)

            if len(entropy_hist) != 0:
                ent_diff = ent - entropy_hist[-1]
                # print(ent_diff)
                # group['lr'] = lr * ent_diff

            entropy_hist.append(ent)

    def get_entropy_history(self):
        entropy_history = {}
        for l, group in enumerate(self.param_groups): # Each group is a layer
            entropy_history[l] = group['entropy_hist']
        return entropy_history

    def print_stuff(self):
        for group in self.param_groups: # Each group is a layer
            lr = group['lr']
            entropy_hist = group['entropy_hist']
            # print(entropy_hist)