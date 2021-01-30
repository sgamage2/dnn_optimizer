from utility import common
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history):
    print('Plotting training history')
    if history is None or 'train_loss' not in history:
        assert False

    fig = plt.figure()
    common.add_figure_to_save(fig, 'training_history')

    train_loss = history['train_loss']

    plt.plot(train_loss, label='training_loss')

    if 'val_loss' in history:
        val_loss = history['val_loss']
        plt.plot(val_loss, label='validation_loss')

    plt.title("Training history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')

    # history_df = pd.DataFrame(history_dict)
    # common.add_dataframe_to_save(history_df, 'training_history')


def plot_entropy_history(history):
    print('Plotting entropy history')
    if history is None or 'entropy_hist' not in history:
        return False

    layer_entropies = history['entropy_hist']

    if type(layer_entropies[0][0]) == np.ndarray:
        plot_entropies_per_neuron(layer_entropies)
    else:
        plot_entropies_per_layer(layer_entropies)


def plot_entropies_per_layer(layer_entropies):
    fig = plt.figure()
    common.add_figure_to_save(fig, 'entropy_history')

    for layer, e_hist in layer_entropies.items():
        epoch = list(range(len(e_hist)))
        plt.plot(e_hist, label='Layer {}'.format(layer))
        # plt.scatter(epoch, entropy_hist)

    # ------------------------------------------------
    # # Plot entropy history from Damith's function for comparison
    # damith_layer_entropies = history['damith_entropies']
    #
    # for layer, damith_entropy_hist in damith_layer_entropies.items():
    #     epoch = list(range(len(damith_entropy_hist)))
    #     plt.plot(damith_entropy_hist, label='Damith Layer {}'.format(layer))
    #     plt.scatter(epoch, damith_entropy_hist, marker='x')
    # ------------------------------------------------

    plt.title("Entropy history")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy (bits)")
    plt.legend(loc='upper right')


def plot_entropies_per_neuron(layer_entropies):
    for layer, e_hist in layer_entropies.items():
        # e_hist is a list of np arrays. Each np array has the entropies of the layer's neurons. Has shape (nodes,)
        neurons = e_hist[0].shape[0]
        fig = plt.figure()
        common.add_figure_to_save(fig, f'layer_{layer}_entropy_history')

        for n in range(neurons):
            neuron_ent_hist = [layer_ent[n] for layer_ent in e_hist]
            plt.plot(neuron_ent_hist)

        plt.title(f"Layer {layer} entropy history")
        plt.xlabel("Epoch")
        plt.ylabel("Entropy (bits)")
        # plt.legend(loc='upper right')


def plot_lr_history(history):
    print('Plotting learning rate history')
    if history is None or 'learning_rate_hist' not in history:
        return False

    layer_lrs = history['learning_rate_hist']

    if type(layer_lrs[0][0]) == np.ndarray:
        plot_lr_per_neuron(layer_lrs)
    else:
        plot_lr_per_layer(layer_lrs)


def plot_lr_per_layer(layer_lrs):
    fig = plt.figure()
    common.add_figure_to_save(fig, 'lr_history')

    for layer, lr_hist in layer_lrs.items():
        plt.plot(lr_hist, label='Layer {}'.format(layer))

    plt.title("Learning rate history")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend(loc='upper right')


def plot_lr_per_neuron(layer_lrs):
    for layer, lr_hist in layer_lrs.items():
        # lr_hist is a list of np arrays. Each np array has the entropies of the layer's neurons. Has shape (nodes,)
        neurons = lr_hist[0].shape[0]
        fig = plt.figure()
        common.add_figure_to_save(fig, f'layer_{layer}_lr_history')

        # layer_lr_hist = np.array(lr_hist)
        # print('====== layer_lr_hist ===== ')
        # print(np.round(layer_lr_hist, 4))
        # print('========================== ')

        for n in range(neurons):
            neuron_lr_hist = [layer_lr[n] for layer_lr in lr_hist]
            # print(neuron_lr_hist)
            plt.plot(neuron_lr_hist)

        plt.title(f"Layer {layer} learning rate history")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        # plt.legend(loc='upper right')