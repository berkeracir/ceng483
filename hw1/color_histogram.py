import sys
import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def usage():
    sys.stderr.write("Usage: python " + sys.argv[0] + " img.jpg\n")
    sys.exit(1)

def color_histogram(image, level, bin):
    """ 3D color histogram
    image = np.array of a colored image file
    level = spatial levels of an image
    bin   = number of bins in the histogram
    """

    if level < 1 or level > 6:
        sys.stderr.write("3D Color Histogram: Level range: [1,6]")
        sys.exit(1)
    if bin < 1 or bin > 256:
        sys.stderr.write("3D Color Histogram: Bin range: [1,256]")
        sys.exit(1)

    histogram = np.zeros((pow(2,level-1)**2, bin, bin, bin), dtype=int)

    bin_range = 256/bin

    intervals = pow(2,level-1)
    x_interval = int(image.shape[0]/intervals)
    y_interval = int(image.shape[1]/intervals)

    k = 0

    for y_level in range(intervals):
        for x_level in range(intervals):
            for y in range(y_level*y_interval, (y_level+1)*y_interval):
                for x in range(x_level*x_interval, (x_level+1)*x_interval):
                    blue_index = int(image[x][y][0]/bin_range)
                    green_index = int(image[x][y][1]/bin_range)
                    red_index = int(image[x][y][2]/bin_range)
                    histogram[k][blue_index][green_index][red_index] += 1
            k += 1

    # Normalize the histograms
    return histogram*intervals**2/(image.shape[0]*image.shape[1])

def plot_color_histogram(histogram): # TODO: adjust for levels
    """ Plots 3D color histogram
    histogram = 2D np.array, the first dimension is for level and the second one
                is for normalized histogram counts
    """

    level = int(math.log(math.sqrt(histogram.shape[0]), 2) + 1)
    bin = histogram.shape[1]

    fig = plt.figure("Level:" + str(level) + " Bin:" + str(bin))

    print("Level:" + str(level) + " Bin:" + str(bin))

    blue_axis = np.zeros(bin**3)
    green_axis = np.zeros(bin**3)
    red_axis = np.zeros(bin**3)
    intensity = np.zeros(bin**3)
    for i in range(bin):
        for j in range(bin):
            for k in range(bin):
                intensity[i*bin**2+j*bin+k] = histogram[0][i][j][k]
                blue_axis[i*bin**2+j*bin+k] = i
                green_axis[i*bin**2+j*bin+k] = j
                red_axis[i*bin**2+j*bin+k] = k

    for i in range(pow(2,level-1)**2):
        ax = fig.add_subplot(111, projection='3d')
        plot3d = ax.scatter(blue_axis, green_axis, red_axis, c=intensity)
        plt.colorbar(plot3d)
        #plt.bar(np.arange(bin), histogram[i][0])
        #plt.plot(histogram[i])

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2: usage()

    img = cv2.imread(sys.argv[1])

    hist = color_histogram(img, 1, 4)
    plot_color_histogram(hist)