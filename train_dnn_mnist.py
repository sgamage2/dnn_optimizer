import torch
from utility import data_prep
from models.cnn_mnist import CNN_Network
from models.model_utility import train_network, test_network


params = {}
params['data_dir'] = 'data'
params['device'] = 'cuda'    # cpu, cuda
params['epochs'] = 5
params['batch_size'] = 64


def main():
    # ------------------------
    # Init
    if params['device'] == 'cuda':
        assert torch.cuda.is_available()
    device = torch.device(params['device'])

    # ------------------------
    # Load data
    trainloader, testloader = data_prep.get_mnist_data_loaders(params['data_dir'], params['batch_size'])
    # display_images(trainloader)

    # ------------------------
    # Create and train model
    cnn_net = CNN_Network()
    train_network(cnn_net, trainloader, params['epochs'], device)

    # ------------------------
    # Test model
    test_network(cnn_net, testloader, device)


if __name__ == '__main__':
    main()
