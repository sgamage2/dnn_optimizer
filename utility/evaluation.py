from utility import common
import matplotlib.pyplot as plt


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
    if history is None or 'per_layer_entropy' not in history:
        return False

    fig = plt.figure()
    common.add_figure_to_save(fig, 'entropy_history')

    layer_entropies = history['per_layer_entropy']

    for layer, entropy_hist in layer_entropies.items():
        epoch = list(range(len(entropy_hist)))
        plt.plot(entropy_hist, label='Layer {}'.format(layer))
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


def plot_lr_history(history):
    print('Plotting learning rate history')
    if history is None or 'per_layer_learning_rate' not in history:
        return False

    fig = plt.figure()
    common.add_figure_to_save(fig, 'lr_history')

    layer_lrs = history['per_layer_learning_rate']

    for layer, lr_hist in layer_lrs.items():
        plt.plot(lr_hist, label='Layer {}'.format(layer))

    plt.title("Learning rate history")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend(loc='upper right')
