import torch
from utility import data_prep
from models.cnn_mnist import CNN_Network
from models.ann_fixed import FixedFullyConnectedNetwork
from models.ann import FullyConnectedNetwork
from models.model_utility import train_network, test_network, visualize_network


params = {}
params['data_dir'] = 'data'
params['device'] = 'cpu'    # cpu, cuda
params['epochs'] = 20
params['batch_size'] = 256

params['ann_input_nodes'] = 28 * 28
params['output_nodes'] = 10
# params['ann_layer_units'] = [64, 32, 16]
params['ann_layer_units'] = [128, 64]
params['ann_layer_activations'] = ['relu', 'relu', 'relu']
params['ann_layer_dropout_rates'] = [0.2, 0.2, 0.2]


def main():
    # ------------------------
    # Init
    assert params['device'] in ['cpu', 'cuda']
    if params['device'] == 'cuda':
        print('Running on GPU (CUDA)')
        assert torch.cuda.is_available()
    elif params['device'] == 'cpu':
        print('Running on CPU')
    device = torch.device(params['device'])

    # ------------------------
    # Load data
    trainset, testset = data_prep.get_mnist_preprocessed(params['data_dir'], params['batch_size'])
    # display_images(trainloader)

    # ------------------------
    # Create and train model
    # cnn_net = CNN_Network()
    # cnn_net = FixedFullyConnectedNetwork(input_size=28*28, output_size=10)
    cnn_net = FullyConnectedNetwork(params)
    visualize_network(cnn_net, trainset)
    train_network(cnn_net, trainset, params['epochs'], params['batch_size'], device)

    # ------------------------
    # Test model
    test_network(cnn_net, testset, device)


if __name__ == '__main__':
    main()
