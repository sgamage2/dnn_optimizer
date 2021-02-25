""" A simple implementation of the Stochastic Gradient Descent (SGD): without weight decay, momentum etc
Follows the PyTorch SGD source code
https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
 """

import numpy as np
import torch
from torch.optim import Optimizer
from utility import entropy

# ToDo: optimization - do all computations (entropy, entropy diff, lr update coeff etc) with torch
# This keeps doing the computations on the same given device (eg: GPU)

class EntropyPerLayer(Optimizer):
    def __init__(self, params, lr, beta,ent_c,ent_k):
        assert lr > 0
        defaults = dict(lr=lr)
        super(EntropyPerLayer, self).__init__(params, defaults)

        for group in self.param_groups:
            group['entropy_hist'] = []
            group['lr_hist'] = []   # Keep a history of learning rates (for debugging)
            group['beta'] = beta
            group['original_lr'] = lr
            group['ent_c'] = ent_c
            group['ent_k'] = ent_k

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

    def on_epoch_start(self):
        """Performs things such as updating learning rate based on entropy.
        This method should be called in the training loop at the start of every epoch
        """
        # The params are in Parameter objects. Each layer has 2 Parameter objects (biases and weights)
        for group in self.param_groups:  # Each group is a layer
            entropy_hist = group['entropy_hist']
            lr_hist = group['lr_hist']

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

            # print(f'---- lrs: {np.round(ent_per_neuron[0:5], 4)}')
            # print(f'---- lrs: {np.round(lr_per_neuron_arr[0:5], 4)}')
            entropy_hist.append(ent)
            lr_hist.append(group['lr'])

        if len(entropy_hist) >= 3:  # To get entropy diff, we need a history of at least 2
            self._update_learning_rates()

    def get_history(self):
        entropy_history = {}
        lr_history = {}
        for l, group in enumerate(self.param_groups): # Each group is a layer
            entropy_history[l] = group['entropy_hist']
            lr_history[l] = group['lr_hist']
        return entropy_history, lr_history

    def _update_learning_rates(self):
        num_layers = len(self.param_groups)
        en_diff_sum = 0.0

        # Iterate over layers to compute average entropy
        for group in self.param_groups:  # Each group is a layer
            entropy_hist = group['entropy_hist']
            en_diff = abs(entropy_hist[-1] - 2*entropy_hist[-2]+entropy_hist[-3])/2  # Diff between last 2 values
            en_diff_sum += en_diff

        epsilon = 1e-4
        avg_en_diff = (en_diff_sum + epsilon) / num_layers

        # Iterate over layers to update learning rates
        for group in self.param_groups:  # Each group is a layer
            entropy_hist = group['entropy_hist']
            beta = group['beta']
            ent_k = group['ent_k']
            ent_c = group['ent_c']
            en_diff = abs(entropy_hist[-1] - 2*entropy_hist[-2]+entropy_hist[-3])/2
            # en_diff = (en_diff + avg_en_diff) / avg_en_diff
            en_diff = en_diff / avg_en_diff
            # coeff = (en_diff * beta) / (1 + en_diff * beta)
            coeff = (2 / (1 + np.exp(-en_diff * beta))-1)*ent_k
            group['lr'] = group['original_lr'] * coeff+ ent_c  # Learning rate update
