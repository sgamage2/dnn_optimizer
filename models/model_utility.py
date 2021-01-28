import time
import torch.nn as nn
import torch.optim as optim
import torch
import torchviz
import optimizers.simple_sgd
import optimizers.per_weight_lr
import optimizers.per_neuron_lr


def train_network(network, trainset, params, device):
    epochs = params['epochs']
    batch_size = params['batch_size']
    optimizer_name = params['optimizer']
    learning_rate = params['learning_rate']

    t0 = time.time()
    X_train = trainset.tensors[0]
    y_train = trainset.tensors[1]

    # Move model and full dataset to device upfront (may fail for large models and datasets)
    network.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_name, network, learning_rate)

    # Some variables for stat printing
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size
    batch_interval = num_batches // 1

    train_loss_hist = []

    print('Training neural network')
    for epoch in range(epochs):  # loop over the dataset multiple epochs
        shuffled_indices = torch.randperm(num_samples)
        train_loss_epoch = 0.0
        for j in range(0, num_samples, batch_size):
            indices = shuffled_indices[j: j + batch_size]
            inputs = X_train[indices]
            labels = y_train[indices]

            optimizer.zero_grad()   # To prevent gradient accumulation

            # forward + backward + optimize
            outputs = network(inputs)
            train_loss_batch = criterion(outputs, labels)
            train_loss_batch.backward()
            optimizer.step()    # Apply optimizer rule and update network params (using the param gradients saved after the backward() call)

            train_loss_epoch += train_loss_batch.item()

            # print statistics
            # running_loss += train_loss_batch.item()
            # if j % batch_interval == 0:  # print every batch_interval mini-batches
            #     print('[epoch: %d, batch: %5d] train_loss_batch: %.3f' %
            #           (epoch + 1, j + 1, running_loss / 2000))
            #     running_loss = 0.0

        train_loss_epoch = float(train_loss_epoch / num_batches)
        train_loss_hist.append(train_loss_epoch)

        print('[epoch: {}] train_loss: {:.3f}'.format(epoch + 1, train_loss_epoch))

    history = {'train_loss': train_loss_hist}

    time_to_train = time.time() - t0
    print('Finished Training. Time taken: {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    return history


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
    print(network)
    X_all = dataset.tensors[0]
    out = network(X_all[0])
    torchviz.make_dot(out).render("output/network_viz", format="png")


def create_optimizer(optimizer_name, network, learning_rate):
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
        assert False
    elif optimizer_name == 'entropy_per_neuron':
        assert False
    else:
        assert False    # Unknown optimizer_name

    return optimizer
