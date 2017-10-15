from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import matplotlib.pyplot as plt


def main():
    ds = Dataset(load_tags=True, filter_tag = True)
    icm = ds.build_icm()
    # get a column vector with the sum of each attribute
    # for each attribute its frequency in the icm
    width = 0.35       # the width of the bars
    tag_freq = ds.tag_counter
    x = np.arange(len(tag_freq))  # location of bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, tag_freq, width, color='r')
    plt.show()


if __name__ == '__main__':
    main()
