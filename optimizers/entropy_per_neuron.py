""" This optimizer keeps a separate learning rate for the set of all incoming weights to a neuron
The learning rates are stored in Tensors of shape of Parameters (weights and biases of each layer)
 """


import torch
from torch.optim import Optimizer
import numpy as np
from utility import entropy
from pprint import pprint

# ToDo: optimization - do all computations (entropy, entropy diff, lr update coeff etc) with torch
# This keeps doing the computations on the same given device (eg: GPU)

class EntropyPerNeuron(Optimizer):
    def __init__(self, params, lr, beta):
        assert lr > 0
        defaults = dict(lr=lr)
        super(EntropyPerNeuron, self).__init__(params, defaults)

        self.device = None  # Needed when creating tensors in LR updates

        for group in self.param_groups:
            group['entropy_hist'] = []    # Each element is an np array with entropies of neurons. shape = (layer_nodes,)
            group['lr_hist'] = []    # Each element is an np array with learning rates of neurons. shape = (layer_nodes,)
            group['beta'] = beta
            # group['original_lr'] = lr
            for p in group['params']:
                state = self.state[p]   # A dict that keeps state of each Parameter obj (biases and weights of layers)
                state['step'] = 0
                state['lr'] = torch.full_like(p, lr)    # A Tensor of same shape as Parameter filled with initial lr
                state['original_lr'] = torch.full_like(p, lr)    # A Tensor of same shape as Parameter filled with initial lr

                if self.device is None:
                    self.device = p.device

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
                assert p.shape == per_weight_lr.shape

                weight_change = -per_weight_lr * d_p    # Tensor of same shape as p and d_p

                # For inplace multiplication, we can do per_weight_lr.mul(-d_p)
                # But this will vary the stored learning rates through the optimization steps

                p.add_(weight_change)      # SGD update rule: p = p - lr * d_p

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
                if len(np_arr.shape) == 1:  # Bias vector
                    np_arr = np_arr.reshape(-1, 1)
                    lr_per_neuron_arr = self.state[p]['lr'].cpu().detach().numpy()
                params_list.append(np_arr)

            # For the connection between two layers L1 --> L2
            # params_list has the bias vector (L2 x 1) and weights matrix (L2 x L1) --> concatenate cols
            # if len(params_list) > 0:
            weights = np.concatenate(params_list, axis=1)   # L2 x (L1 + 1)

            ent_per_neuron = self._compute_per_neuron_entropies(weights)    # Has shape (L2,)

            # print(f'---- lrs: {np.round(ent_per_neuron[0:5], 4)}')
            # print(f'---- lrs: {np.round(lr_per_neuron_arr[0:5], 4)}')
            entropy_hist.append(ent_per_neuron)
            lr_hist.append(np.round(lr_per_neuron_arr, 5))  # The rounding down is important to store the value correctly
            # print(f'---- lr_hist[-1]: {np.round(lr_hist[-1], 4)}')
            # pprint(lr_hist)

        if len(entropy_hist) >= 2:  # To get entropy diff, we need a history of at least 2
            self._update_learning_rates()

    def get_history(self):
        entropy_history = {}
        lr_history = {}
        for l, group in enumerate(self.param_groups): # Each group is a layer
            entropy_history[l] = group['entropy_hist']
            lr_history[l] = group['lr_hist']
        return entropy_history, lr_history

    def _compute_per_neuron_entropies(self, weights):
        # For the connection between two layers L1 --> L2
        # weights has shape L2 x (L1 + 1)
        L2_nodes = weights.shape[0]
        ent_per_neuron = np.zeros(L2_nodes)
        for row in range(L2_nodes):
            neuron_weights = weights[row, :]
            ent_per_neuron[row] = entropy.get_entropy(neuron_weights)
        return ent_per_neuron   # Has shape (L2,)

    def _update_learning_rates(self):
        num_nodes = 0
        en_diff_sum = 0.0

        # Iterate over layers to compute average entropy
        for group in self.param_groups:  # Each group is a layer
            entropy_hist = group['entropy_hist']
            en_diff = np.abs(entropy_hist[-1] - entropy_hist[-2])  # Diff between last 2 ent values, shape = (L2,)
            en_diff_sum += np.sum(en_diff)
            num_nodes += en_diff.size

        epsilon = 1e-5
        avg_en_diff = (en_diff_sum + epsilon) / num_nodes   # scalar

        # Iterate parameters in each layer to update learning rates
        for group in self.param_groups:  # Each group is a layer
            entropy_hist = group['entropy_hist']
            beta = group['beta']

            en_diff = abs(entropy_hist[-1] - entropy_hist[-2])  # shape = (L2,)
            en_diff = (en_diff + avg_en_diff) / avg_en_diff
            coeffs = (en_diff * beta) / (1 + en_diff * beta)   # shape = (L2,)

            coeffs = torch.from_numpy(coeffs).float().to(self.device)
            coeffs_reshaped = coeffs.reshape(-1, 1) # Reshape needed for broadcast when multiplying with matrix

            # print(f'coeffs.shape: {coeffs.shape}')
            # print(f'coeffs: {coeffs[0:8]}')

            for p in group['params']:  # p = params. A Parameter object with biases or weights of a layer
                state = self.state[p]
                # per_weight_lr = state['lr']
                assert p.shape == state['lr'].shape

                # print(f'--- p.shape: {p.shape}')

                # For the connection between two layers L1 --> L2
                if p.ndim == 1:  # This Parameter is a vector of biases (shape = L2_nodes)
                    state['lr'] = state['original_lr'] * coeffs
                    # print(f'---- before ---- {per_weight_lr[0:8]}')
                    # per_weight_lr.mul_(coeffs)
                    # print(f'---- after ---- {per_weight_lr[0:8]}')
                    # print(f'---- after ---- {state["lr"][0:8]}')
                elif p.ndim == 2:  # This Parameter is a Tensor of weights (shape = L2_nodes, L1_nodes)
                    state['lr'] = state['original_lr'] * coeffs_reshaped
                    # per_weight_lr.mul_(coeffs_reshaped)

