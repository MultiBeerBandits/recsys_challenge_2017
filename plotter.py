from loader_v2 import *
from scipy.sparse import *
import numpy as np
import matplotlib.pyplot as plt


def main():
    ds = Dataset()
    icm = ds.build_icm()
    # get a column vector with the sum of each attribute
    # for each attribute its frequency in the icm
    width = 0.35       # the width of the bars
    print("Attributes :", icm.shape[0])
    attr_freq = np.ravel(icm.sum(axis=1))
    x = np.arange(len(attr_freq))  # location of bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, attr_freq, width, color='r')
    plt.show()


if __name__ == '__main__':
    main()
