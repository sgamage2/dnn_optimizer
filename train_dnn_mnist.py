# Run this script with the following command
# python train_dnn_mnist.py sunanda_params.yaml

import torch
# import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from pprint import pformat

from utility import data_prep, evaluation, common
from models.cnn_mnist import CNN_Network
from models.ann_fixed import FixedFullyConnectedNetwork
from models.ann import FullyConnectedNetwork
from models.model_utility import train_network, test_network, visualize_network


def run_experiment(params_filename):
    # ------------------------
    # Init
    torch.manual_seed(123)
    params = common.get_params_from_file(params_filename)
    common.init_logging(params['output_dir'])

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

    history = train_network(cnn_net, trainset, testset, params, device)

    # ------------------------
    # Visualization
    evaluation.plot_training_history(history)
    evaluation.plot_entropy_history(history)
    evaluation.plot_lr_history(history)

    # ------------------------
    # Test model
    test_network(cnn_net, testset, device)

    # plt.show()
    common.save_all_figures(params['output_dir'])


def main():
    params_filename = common.get_params_filename_from_cmd_args()
    run_experiment(params_filename)


if __name__ == '__main__':
    main()
