import torch
# import matplotlib.pyplot as plt

from utility import data_prep, evaluation, common
from models.cnn_mnist import CNN_Network
from models.ann_fixed import FixedFullyConnectedNetwork
from models.ann import FullyConnectedNetwork
from models.model_utility import train_network, test_network, visualize_network
from pprint import pformat


params = {}
params['data_dir'] = 'data'
params['device'] = 'cpu'    # cpu, cuda
params['epochs'] = 10
params['batch_size'] = 20 # 256
params['optimizer'] = 'entropy_per_layer'     # sgd, sgd_momentum, adagrad, adam, entropy_per_layer, entropy_per_neuron ...
params['learning_rate'] = 0.1
params['ent_beta'] = 1  # Param for entropy_per_layer optimizer (for learning rule update)

params['ann_input_nodes'] = 28 * 28
params['output_nodes'] = 10
# params['ann_layer_units'] = [64, 32, 16]
# params['ann_layer_units'] = [128, 64]
params['ann_layer_units'] = [50, 50]
params['ann_layer_activations'] = ['relu', 'relu', 'relu']
# params['ann_layer_dropout_rates'] = [0.2, 0.2, 0.2]
params['ann_layer_dropout_rates'] = [0.0, 0.0, 0.0]


def main():
    # ------------------------
    # Init
    torch.manual_seed(123)

    print('Experiment parameters')
    print(pformat(params))

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

    history = train_network(cnn_net, trainset, params, device)

    # ------------------------
    # Visualization
    evaluation.plot_training_history(history)
    evaluation.plot_entropy_history(history)
    evaluation.plot_lr_history(history)

    # ------------------------
    # Test model
    test_network(cnn_net, testset, device)

    # plt.show()
    common.save_all_figures('output')


if __name__ == '__main__':
    main()
