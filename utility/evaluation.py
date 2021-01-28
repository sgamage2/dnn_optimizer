from utility import common
import matplotlib.pyplot as plt


def plot_training_history(history):
    print('Plotting history')
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

