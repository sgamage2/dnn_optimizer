from utility import entropy
import numpy as np


def entropy_test():
    for i in range(3):  # repeat 3 times
        weights = np.random.randn(64, 32)
        ed = entropy.get_entropy_deprecated(weights)
        e = entropy.get_entropy(weights)
        print('entropy_deprecated: {}, entropy_optimized: {}'.format(ed, e))
        assert np.isclose(ed, e)


def main():
    entropy_test()


if __name__ == '__main__':
    main()
