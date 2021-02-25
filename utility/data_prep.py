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

""" Loads MNIST into a dataloader. Each batch is preprocessed in the training loop: very expensive """
def get_mnist_data_loaders(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),     # Convert PIL image to tensor
         transforms.Normalize((0.5), (0.5))])   # (mean1, mean2, mean3), (std1, std2, std3) for the 3 channels

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader


""" Loads MNIST, preprocesss and load into a dataloader. Training loop has no preprocessing steps: fast """
def get_mnist_preprocessed(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),     # Convert PIL image to tensor
         transforms.Normalize((0.5), (0.5)),       # (mean1, mean2, mean3), (std1, std2, std3) for the 3 channels
         ])

    print('Preprocessing MNIST training set')
    trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    preprocessed_trainset = preprocess_mnist(trainset)
    # trainloader = torch.utils.data.DataLoader(preprocessed_trainset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Preprocessing MNIST test set')
    testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    preprocessed_testset = preprocess_mnist(testset)
    # testloader = torch.utils.data.DataLoader(preprocessed_testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # return trainloader, testloader
    return preprocessed_trainset, preprocessed_testset

""" Loads MNIST, preprocesss and load into a dataloader. Training loop has no preprocessing steps: fast """
def get_fashion_mnist_preprocessed(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),     # Convert PIL image to tensor
         transforms.Normalize((0.5), (0.5)),       # (mean1, mean2, mean3), (std1, std2, std3) for the 3 channels
         ])

    print('Preprocessing MNIST training set')
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    preprocessed_trainset = preprocess_mnist(trainset)
    # trainloader = torch.utils.data.DataLoader(preprocessed_trainset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Preprocessing MNIST test set')
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    preprocessed_testset = preprocess_mnist(testset)
    # testloader = torch.utils.data.DataLoader(preprocessed_testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # return trainloader, testloader
    return preprocessed_trainset, preprocessed_testset

def preprocess_mnist(dataset):
    batch_size = dataset.data.shape[0]  # To get a single batch that contains full dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataiter = iter(dataloader)
    x_preprocessed, y = dataiter.next()  # A single batch that contains full dataset
    assert x_preprocessed.shape[0] == batch_size  # Verify that this single batch contains the full dataset

    preprocessed_dataset = torch.utils.data.TensorDataset(x_preprocessed, y)
    return preprocessed_dataset


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

