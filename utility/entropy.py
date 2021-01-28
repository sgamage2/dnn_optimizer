import numpy as np


# Originally written by Damith
def get_entropy_deprecated(weights):
    n = weights.size
    hist = np.histogram(weights, bins=30)  # getting the histogram
    # print(hist)
    hist_1 = [x / n for x in hist[0]]

    entropy = 0
    for x in hist_1:
        if x > 0:
            entropy -= x * np.log(np.abs(x))

    return entropy


# Optimized version of the entropy function
# ToDo: Implement this with pytorch functions (torch.histc and torch.log)
def get_entropy(weights):
    weights = np.ravel(weights)
    weights = weights[~np.isnan(weights)]   # Drop possible NaN values

    if len(weights) == 0:
        return 0

    # print(weights)

    n = weights.size
    hist = np.histogram(weights, bins=30)  # getting the histogram
    counts_normalized = hist[0] / n

    x = counts_normalized[counts_normalized > 0]
    # x = x[np.isfinite(x)]
    e = - x * np.log(x)
    entropy = np.sum(e)
    return entropy


def get_entropies_damith(net, epoch):
    entropies = []
    layer = 0
    for layer_cont in net.layers:
        # print(f'layer: {layer}')
        # inside a layer get details
        total_data = []  # get all the parameters of a particular layer
        for layer_params in layer_cont.parameters():
            # get parameters inside a layer
            for q in range(len(layer_params)):
                # print(layer_params[q].data.numpy().shape)
                # print (layer_params[q].data.numpy())

                # collect parameters of the layer
                z = layer_params[q].data.numpy().tolist()
                if (isinstance(z, list)):
                    total_data += [a for a in z]
                else:  # Bias of each node as scalar
                    # print(z)
                    # assert False   # Debugging
                    total_data += [z]


        if (len(total_data) > 0):
            # print("Length of layer = ", len(total_data))
            hist = np.histogram(np.array(total_data), bins=30)  # getting the histogram
            # print(hist)
            hist_1 = [x / len(total_data) for x in hist[0]]
            entropy = 0

            for x in hist_1:
                if x > 0:
                    entropy -= x * np.log(np.abs(x))

            entropies += [entropy]

        layer += 1

    return entropies
