import sys
import math
import numpy as np
import cv2
from PIL import Image
import scipy.signal
import matplotlib.pyplot as plt
import time

def usage():
    sys.stderr.write("Usage: python " + sys.argv[0] + " img.jpg\n")
    sys.exit(1)

def rad2deg(rad):
    retval = math.degrees(rad)

    if retval < 0:
        retval += 360

    return retval

def gradient_histogram(image, vertical_filter, horizontal_filter, level, bin):
    """ Gradient histogram
    image = np.array of an image file
    vertical_filter   = vertical (y) derivative filter
    horizontal_filter = horizontal (x) derivative filter
    level = spatial levels of an image
    bin   = number of bins in the histogram
    """

    if level < 1 or level > 6:
        sys.stderr.write("Grayscale Histogram: Level range: [1,6]\n")
        sys.exit(1)
    if bin < 1 or bin > 360:
        sys.stderr.write("Grayscale Histogram: Bin range: [1,360]\n")
        sys.exit(1)

    histogram = np.zeros((pow(2,level-1)**2, bin))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vertical = scipy.signal.convolve2d(gray, vertical_filter, 
                                      mode='same', boundary='fill', fillvalue=0)
    horizontal = scipy.signal.convolve2d(gray, horizontal_filter, 
                                      mode='same', boundary='fill', fillvalue=0)

    bin_range = 360/bin

    intervals = pow(2,level-1)
    y_interval = int(gray.shape[0]/intervals)
    x_interval = int(gray.shape[1]/intervals)

    k = 0

    for y_level in range(intervals):
        for x_level in range(intervals):
            for y in range(y_level*y_interval, (y_level+1)*y_interval):
                for x in range(x_level*x_interval, (x_level+1)*x_interval):
                    v = vertical[y][x]
                    h = horizontal[y][x]
                    orientation = rad2deg(math.atan2(v, h))
                    magnitude = math.sqrt(v**2 + h**2)
                    bin_index = int(orientation/bin_range)
                    histogram[k][bin_index] += magnitude
            # Normalize the histogram
            histogram[k] = histogram[k]/sum(histogram[k])
            k += 1

    return histogram

def plot_gradient_histogram(histogram):
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

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2: usage()

    img = cv2.imread(sys.argv[1])
    horizontal_kernel = np.array([[-1, 0, 1], 
                                  [-2, 0, 2], 
                                  [-1, 0, 1]])

    vertical_kernel = np.array([[1, 2, 1], 
                                [0, 0, 0], 
                                [-1, -2, -1]])

    hist = gradient_histogram(img, vertical_kernel, horizontal_kernel, 1, 16)
    plot_gradient_histogram(hist)

    """
    kernel_v2 = np.array([[2, 1, 0, -1, -2], 
                          [2, 1, 0, -1, -2],
                          [4, 2, 0, -2, -4], 
                          [2, 1, 0, -1, -2],
                          [2, 1, 0, -1, -2]])

    kernel_h2 = np.array([[2, 2, 4, 2, 2],
                          [1, 1, 2, 1, 1], 
                          [0, 0, 0, 0, 0],
                          [-1, -1, -2, -1, -1],
                          [-2, -2, -4, -2, -2]])
    """