import torch
# import matplotlib.pyplot as plt

from utility import data_prep, evaluation, common
from models.cnn_mnist import CNN_Network
from models.ann_fixed import FixedFullyConnectedNetwork
from models.ann import FullyConnectedNetwork
from models.model_utility import train_network, test_network, visualize_network
from pprint import pformat
import sys, os
import yaml


def get_params_filename_from_cmd_args():
    num_args = len(sys.argv)
    if num_args != 2:
        print('Usage:')
        print('python train_dnn_mnist my_config.yaml')
        sys.exit(1)

    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print(f'Config File {filename} does not exist')
        sys.exit()

    return filename


def run_experiment(params_filename):
    # ------------------------
    # Init
    torch.manual_seed(123)

    # Load params
    file = open(params_filename, "r")
    params = yaml.load(file, Loader=yaml.FullLoader)

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
    params_filename = get_params_filename_from_cmd_args()
    run_experiment(params_filename)


if __name__ == '__main__':
    main()
