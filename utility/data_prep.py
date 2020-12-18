import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def get_cifar10_data_loaders(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),     # Convert PIL image to tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # (mean1, mean2, mean3), (std1, std2, std3) for the 3 channels

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def get_mnist_data_loaders(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),     # Convert PIL image to tensor
         transforms.Normalize((0.5), (0.5))])   # (mean1, mean2, mean3), (std1, std2, std3) for the 3 channels

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def display_images(data_loader):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    # print(images.shape)
    # imshow(torchvision.utils.make_grid(images))

