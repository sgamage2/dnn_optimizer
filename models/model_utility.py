import time
import torch.nn as nn
import torch.optim as optim
import torch
import torchviz
import optimizers.simple_sgd
import optimizers.per_weight_lr
import optimizers.entropy_per_neuron
import optimizers.entropy_per_layer
from utility import entropy
from pprint import pprint


def train_network(network, trainset, testset, params, device):
    epochs = params['epochs']
    batch_size = params['batch_size']
    optimizer_name = params['optimizer']

    t0 = time.time()
    X_train = trainset.tensors[0]
    y_train = trainset.tensors[1]
    X_test = testset.tensors[0]
    y_test = testset.tensors[1]

    # Move model and full dataset to device upfront (may fail for large models and datasets)
    network.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_name, network, params)

    # History variables
    train_loss_hist = []
    train_error_hist = []
    test_loss_hist = []
    test_error_hist = []
    # damith_ents_hist = {0: [], 1: [], 2: []}

    print('Training neural network')
    for epoch in range(epochs):  # loop over the dataset multiple epochs
        epoch_start_time = time.time()

        train_loss, train_error = train_one_epoch(X_train, y_train, network, optimizer, loss_function, batch_size)
        test_loss, test_error = test_one_epoch(X_test, y_test, network, loss_function)

        train_loss_hist.append(train_loss)
        train_error_hist.append(train_error)
        test_loss_hist.append(test_loss)
        test_error_hist.append(test_error)

        # ------------------------------------------------------
        # # Compute per layer entropy from Damith's function for comparison
        # ents_damith = entropy.get_entropies_damith(network, epoch)
        # # print(f'Damith entropy: {ents_damith}')
        # for layer, ent in enumerate(ents_damith):
        #     damith_ents_hist[layer].append(ent)
        # ------------------------------------------------------

        epoch_info = f'[epoch: {epoch + 1}/{epochs}]\t' \
                     f'time: {time.time() - epoch_start_time:.1f}\t' \
                     f'train_loss: {train_loss:.4f}\t' \
                     f'train_error: {train_error * 100:.2f}%\t' \
                     f'test_loss: {test_loss:.4f}\t' \
                     f'test_error: {test_error * 100:.2f}%'

        print(epoch_info)

    history = {}
    history['train_loss'] = train_loss_hist
    history['train_error'] = train_error_hist
    history['test_loss'] = test_loss_hist
    history['test_error'] = test_error_hist

    if "get_history" in dir(optimizer):     # get_history() method is only available in some of the custom optimizers
        ent_hist, lr_hist = optimizer.get_history()
        history['entropy_hist'] = ent_hist
        history['learning_rate_hist'] = lr_hist
        # history['damith_entropies'] = damith_ents_hist

    # pprint(lr_hist)

    time_to_train = time.time() - t0
    print('Finished Training. Time taken: {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    return history


def train_one_epoch(X_train, y_train, network, optimizer, loss_function, batch_size):
    num_samples = X_train.shape[0]

    optimizer.on_epoch_start()  # Optimizer may do things like compute entropies and update learning rates

    shuffled_indices = torch.randperm(num_samples)
    train_loss_epoch = 0.0
    incorrect_preds = 0
    for j in range(0, num_samples, batch_size):
        indices = shuffled_indices[j: j + batch_size]
        inputs = X_train[indices]
        labels = y_train[indices]

        optimizer.zero_grad()  # To prevent gradient accumulation

        # forward + backward + optimize
        outputs = network(inputs)
        train_loss_batch = loss_function(outputs, labels)
        train_loss_batch.backward()
        optimizer.step()  # Apply optimizer rule and update network params (using the param gradients saved after the backward() call)

        train_loss_epoch += train_loss_batch.item()

        _, labels_pred = torch.max(outputs.data, 1)
        incorrect_preds += (labels_pred != labels).sum().item()

    num_batches = num_samples // batch_size
    train_loss_epoch = float(train_loss_epoch / num_batches)

    train_error = incorrect_preds / num_samples

    return train_loss_epoch, train_error


def test_one_epoch(X_test, y_test, network, loss_function):
    num_samples = X_test.shape[0]

    with torch.no_grad():  # We do not intend to call backward(), as we don't backprop and optimize
        # Assume we can do predictions for the entire test set (may fail for large test sets)
        outputs = network(X_test)
        _, y_pred = torch.max(outputs.data, 1)

        test_loss = loss_function(outputs, y_test)
        incorrect_preds = (y_pred != y_test).sum().item()
        test_error = incorrect_preds / num_samples

    return test_loss, test_error


def test_network(network, testset, device):
    X_test = testset.tensors[0]
    y_test = testset.tensors[1]

    # Move model and full dataset to device upfront (may fail for large models and datasets)
    network.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    num_samples = X_test.shape[0]

    with torch.no_grad():   # We do not intend to call backward(), as we don't backprop and optimize
        # Assume we can do predictions for the entire test set (may fail for large test sets)
        outputs = network(X_test)
        _, y_pred = torch.max(outputs.data, 1)
        correct_preds = (y_pred == y_test).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_preds / num_samples))



def visualize_network(network, dataset):
    print(network.layers)
    print("Number of model parameters = ", sum(p.numel() for p in network.parameters()))
    X_all = dataset.tensors[0]
    out = network(X_all[0])
    torchviz.make_dot(out).render("output/network_viz", format="png")


def create_optimizer(optimizer_name, network, params):
    learning_rate = params['learning_rate']
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=learning_rate)  # SGD without momentum
    elif optimizer_name == 'momentum':
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(network.parameters())
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(network.parameters())
    elif optimizer_name == 'sgd_per_layer': # Only supports a 3-layer network. ToDo: Make this generic
        # Create parameter groups and assign learning rates to them
        param_groups = network.group_params_by_layer()
        layer_lrs = [learning_rate * 10, learning_rate, learning_rate * 0.1]
        for group, lr in zip(param_groups, layer_lrs):
            group['lr'] = lr
        optimizer = optimizers.simple_sgd.SimpleSGD(param_groups, lr=learning_rate)
    elif optimizer_name == 'entropy_per_layer':
        param_groups = network.group_params_by_layer()
        for group in param_groups:
            group['lr'] = learning_rate
        beta = params['ent_beta']
        optimizer = optimizers.entropy_per_layer.EntropyPerLayer(param_groups, lr=learning_rate, beta=beta)
    elif optimizer_name == 'entropy_per_neuron':
        param_groups = network.group_params_by_layer()
        for group in param_groups:
            group['lr'] = learning_rate
        beta = params['ent_beta']
        optimizer = optimizers.entropy_per_neuron.EntropyPerNeuron(param_groups, lr=learning_rate, beta=beta)
    else:
        assert False    # Unknown optimizer_name

    return optimizer
