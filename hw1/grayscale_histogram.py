import sys
import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time

def usage():
    sys.stderr.write("Usage: python " + sys.argv[0] + " img.jpg\n")
    sys.exit(1)

def grayscale_histogram(image, level, bin):
    """ Grayscale histogram
    image = np.array of an image file
    level = spatial levels of an image
    bin   = number of bins in the histogram
    """

    if level < 1 or level > 6:
        sys.stderr.write("Grayscale Histogram: Level range: [1,6]")
        sys.exit(1)
    if bin < 1 or bin > 256:
        sys.stderr.write("Grayscale Histogram: Bin range: [1,256]")
        sys.exit(1)

    histogram = np.zeros((pow(2,level-1)**2, bin), dtype=int)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bin_range = 256/bin

    intervals = pow(2,level-1)
    x_interval = int(gray.shape[0]/intervals)
    y_interval = int(gray.shape[1]/intervals)

    k = 0

    for y_level in range(intervals):
        for x_level in range(intervals):
            for y in range(y_level*y_interval, (y_level+1)*y_interval):
                for x in range(x_level*x_interval, (x_level+1)*x_interval):
                    bin_index = int(gray[x][y]/bin_range)
                    histogram[k][bin_index] += 1
            k += 1

    # Normalize the histograms
    return histogram*intervals**2/gray.size

def plot_grayscale_histogram(histogram):
    """ Plots grayscale histogram
    histogram = 2D np.array, the first dimension is for level and the second one
                is for normalized histogram counts
    """

    level = int(math.log(math.sqrt(histogram.shape[0]), 2) + 1)
    bin = histogram.shape[1]

    plt.figure("Level:" + str(level) + " Bin:" + str(bin))

    print("Level:" + str(level) + " Bin:" + str(bin))

    for i in range(pow(2,level-1)**2):
        plt.subplot(pow(2,level-1), pow(2,level-1), i+1)
        plt.bar(np.arange(bin), histogram[i])
        #plt.plot(histogram[i])

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2: usage()

    img = cv2.imread(sys.argv[1])
    hist256 = grayscale_histogram(img, 3, 16)

    plot_grayscale_histogram(hist256)