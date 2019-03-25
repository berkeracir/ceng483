import os
import sys
import cv2
import math
import numpy as np

import color_histogram as color_hist
import gradient_histogram as grad_hist
import grayscale_histogram as gray_hist

DATASET_PATH = "dataset"
CACHE_PATH = "cache"

def usage():
    sys.stderr.write("Usage: python " + sys.argv[0] + " image_dataset.dat\n")
    sys.exit(1)

def squeeze_histogram(histogram):
    if histogram.shape[1] == 1:
        return histogram

    if len(histogram.shape) == 2:
        level = histogram.shape[0]
        bin = histogram.shape[1]
        
        retval = np.zeros((level, int(bin/2)))

        for i in range(bin):
            retval[0][int(i/2)] += histogram[0][i]

        return retval
    else:
        level = histogram.shape[0]
        bin = histogram.shape[1]
        
        retval = np.zeros((level, int(bin/2), int(bin/2), int(bin/2)))

        for i in range(bin):
            for j in range(bin):
                for k in range(bin):
                    retval[0][int(i/2)][int(j/2)][int(k/2)] += histogram[0][i][j][k]

        return retval

def precompute_grayscale_histograms(image):
    img = cv2.imread(os.path.join(DATASET_PATH, image))
    img_name = image.split(".")[0]

    level = 1
    bin = 256

    gray_histogram = gray_hist.grayscale_histogram(img, level, bin)

    for i in range(int(math.log(bin, 2)) + 1):
        conf = "gray_l" + str(level) + "b" + str(bin)

        if not os.path.exists(os.path.join(CACHE_PATH, conf)):
            os.makedirs(os.path.join(CACHE_PATH, conf))

        np.save(os.path.join(CACHE_PATH, conf, img_name), gray_histogram)

        gray_histogram = squeeze_histogram(gray_histogram)
        bin = int(bin/2)

def precompute_color_histograms(image):
    img = cv2.imread(os.path.join(DATASET_PATH, image))
    img_name = image.split(".")[0]

    level = 1
    bin = 64

    color_histogram = color_hist.color_histogram(img, level, bin)

    for i in range(int(math.log(bin, 2)) + 1):
        conf = "color_l" + str(level) + "b" + str(bin)

        if not os.path.exists(os.path.join(CACHE_PATH, conf)):
            os.makedirs(os.path.join(CACHE_PATH, conf))

        np.save(os.path.join(CACHE_PATH, conf, img_name), color_histogram)

        color_histogram = squeeze_histogram(color_histogram)
        bin = int(bin/2)

#def precompute_gradient_histograms(image):

if __name__ == "__main__":
    if len(sys.argv) != 2: usage()

    if not os.path.exists(DATASET_PATH):
        sys.stderr.write("Dataset folder does not exist \""
                        + DATASET_PATH + "\"\n")
        sys.exit(1)

    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    dataset_images = []

    with open(sys.argv[1], "r") as dataset_file:
        for dataset_line in dataset_file:
            img_name = dataset_line.strip("\r\n")
            dataset_images.append(img_name)

    for image in dataset_images:
        #precompute_grayscale_histograms(image) # DONE
        precompute_color_histograms(image)
        #precompute_gradient_histograms(image)

        