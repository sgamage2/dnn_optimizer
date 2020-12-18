import time
import torch.nn as nn
import torch.optim as optim
import torch
import torchviz


def train_network(network, trainloader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    t0 = time.time()

    network.to(device)

    # Some variables for stat printing
    num_samples = trainloader.dataset.data.shape[0]
    batch_size = trainloader.batch_size
    num_batches = num_samples // batch_size
    batch_interval = num_batches // 10

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()   # To prevent gradient accumulation

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    # Apply optimizer rule and update network params (using the param gradients saved after the backward() call)

            # print statistics
            running_loss += loss.item()
            if i % batch_interval == 0:  # print every batch_interval mini-batches
                print('[epoch: %d, batch: %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    time_to_train = time.time() - t0
    print('Finished Training. Time taken: {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))


def test_network(network, testloader, device):
    network.to(device)

    correct = 0
    total = 0
    with torch.no_grad():   # We do not intend to call backward(), as we don't backprop and optimize
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def visualize_network(network, dataloader):
    print(network)
    dataiter = iter(dataloader)
    x_batch, y_batch = dataiter.next()
    out = network(x_batch)
    torchviz.make_dot(out).render("output/network_viz", format="png")
