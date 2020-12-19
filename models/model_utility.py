import time
import torch.nn as nn
import torch.optim as optim
import torch
import torchviz
import optimizers.simple_sgd

def train_network(network, trainset, epochs, batch_size, device):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.SGD(network.parameters(), lr=0.01)  # SGD without momentum (to compare with SimpleSGD)
    # optimizer = optim.Adam(network.parameters())
    optimizer = optimizers.simple_sgd.SimpleSGD(network.parameters(), lr=0.01)

    t0 = time.time()

    X_train = trainset.tensors[0]
    y_train = trainset.tensors[1]

    # Move model and full dataset to device upfront (may fail for large models and datasets)
    network.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Some variables for stat printing
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size
    batch_interval = num_batches // 1

    print('Training neural network')
    for epoch in range(epochs):  # loop over the dataset multiple epochs
        shuffled_indices = torch.randperm(num_samples)
        running_loss = 0.0
        for j in range(0, num_samples, batch_size):
            indices = shuffled_indices[j: j + batch_size]
            inputs = X_train[indices]
            labels = y_train[indices]

            optimizer.zero_grad()   # To prevent gradient accumulation

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    # Apply optimizer rule and update network params (using the param gradients saved after the backward() call)

            # print statistics
            # running_loss += loss.item()
            # if j % batch_interval == 0:  # print every batch_interval mini-batches
            #     print('[epoch: %d, batch: %5d] loss: %.3f' %
            #           (epoch + 1, j + 1, running_loss / 2000))
            #     running_loss = 0.0
        print('[epoch: {}]: loss: :.3f'.format(epoch + 1, loss.item()))

    time_to_train = time.time() - t0
    print('Finished Training. Time taken: {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))


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
