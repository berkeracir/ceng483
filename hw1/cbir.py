import os
import sys
import math
import numpy as np
import cv2
import glob

import color_histogram as color_hist
import gradient_histogram as grad_hist
import grayscale_histogram as gray_hist

DATASET_PATH = "dataset"
OUTPUT_PATH = "out"
CACHE_PATH = "cache"

def usage():
    sys.stderr.write("Usage: python " + sys.argv[0] + " query_images.dat image_dataset.dat\n")
    sys.exit(1)

def euclidean_distance(hist1, hist2):
    if hist1.shape == hist2.shape:
        level = hist1.shape[0]
        bin = hist1.shape[1]

        if len(hist1.shape) > 2:
            color = True
        else:
            color = False
    else:
        sys.stderr.write("Shape mismatch: " + 
                          str(hist1.shape) + " - " + str(hist2.shape) + "\n")
        exit(1)

    d = 0

    if not color:
        for l in range(level):
            for b in range(bin):
                d += (hist1[l][b] - hist2[l][b])**2
    else:
        for l in range(level):
            for b in range(bin):
                for g in range(bin):
                    for r in range(bin):
                        d += (hist1[l][b][g][r] - hist2[l][b][g][r])**2

 
    euclidean_distance = math.sqrt(d)
    return euclidean_distance

if __name__ == "__main__":
    if len(sys.argv) != 3: usage()

    if not os.path.exists(DATASET_PATH):
        sys.stderr.write("Dataset folder does not exist \""
                        + DATASET_PATH + "\"\n")
        sys.exit(1)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    query_images = []
    dataset_images = []

    with open(sys.argv[1], "r") as query_file:
        for query_line in query_file:
            img_name = query_line.strip("\r\n")
            query_images.append(img_name)

    with open(sys.argv[2], "r") as dataset_file:
        for dataset_line in dataset_file:
            img_name = dataset_line.strip("\r\n")
            dataset_images.append(img_name)

    """
    grayscale_levels = [1]
    grayscale_bin_conf = [256, 128, 64, 32, 16, 8, 4, 2]
    
    for level in grayscale_levels:
        for bin in grayscale_bin_conf:
            conf = "gray_l" + str(level) + "b" + str(bin)

            if not os.path.exists(os.path.join(CACHE_PATH, conf)):
                os.path.makedirs(os.path.join(CACHE_PATH, conf))

            fname = conf + ".txt"

            with open(os.path.join(OUTPUT_PATH, fname), "w") as f:
                for query_image in query_images:
                    print(query_image)
                    dist_vector = []

                    qimg_name = query_image.split(".")[0]

                    try:
                        qhist = np.load(os.path.join(CACHE_PATH, conf,
                                                        qimg_name + ".npy"))
                    except FileNotFoundError:
                        qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                        qhist = gray_hist.grayscale_histogram(qimg, level, bin)
                        np.save(os.path.join(CACHE_PATH, conf, qimg_name), qhist)

                    for image in dataset_images:
                        img_name = image.split(".")[0]
                        img = cv2.imread(os.path.join(DATASET_PATH, image))

                        try:
                            hist = np.load(os.path.join(CACHE_PATH, conf,
                                                            img_name + ".npy"))
                        except FileNotFoundError:
                            hist = gray_hist.grayscale_histogram(img, level, bin)
                            np.save(os.path.join(CACHE_PATH, conf, img_name), hist)

                        dist = euclidean_distance(qhist, hist)
                        dist_vector.append((dist, image))

                    sorted_dist_vector = sorted(dist_vector, key=lambda x: x[0])

                    f.write(query_image + ":")
                    for pair in sorted_dist_vector:
                        f.write(" " + str(pair[0]) + " " + pair[1])
                    f.write("\n")
    """

    color_levels = [1]
    color_bin_conf = [2, 4, 8, 16, 32, 64] # [64, 32, 16, 8, 4, 2]

    for level in color_levels:
        for bin in color_bin_conf:
            conf = "color_l" + str(level) + "b" + str(bin)

            if not os.path.exists(os.path.join(CACHE_PATH, conf)):
                os.path.makedirs(os.path.join(CACHE_PATH, conf))

            fname = conf + ".txt"

            with open(os.path.join(OUTPUT_PATH, fname), "w") as f:
                for query_image in query_images:
                    print(query_image)
                    dist_vector = []

                    qimg_name = query_image.split(".")[0]

                    try:
                        qhist = np.load(os.path.join(CACHE_PATH, conf,
                                                        qimg_name + ".npy"))
                    except FileNotFoundError:
                        qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                        qhist = color_hist.color_histogram(qimg, level, bin)
                        np.save(os.path.join(CACHE_PATH, conf, qimg_name), qhist)

                    for image in dataset_images:
                        img_name = image.split(".")[0]
                        img = cv2.imread(os.path.join(DATASET_PATH, image))

                        try:
                            hist = np.load(os.path.join(CACHE_PATH, conf,
                                                            img_name + ".npy"))
                        except FileNotFoundError:
                            hist = color_hist.color_histogram(img, level, bin)
                            np.save(os.path.join(CACHE_PATH, conf, img_name), hist)

                        dist = euclidean_distance(qhist, hist)
                        dist_vector.append((dist, image))

                    sorted_dist_vector = sorted(dist_vector, key=lambda x: x[0])

                    f.write(query_image + ":")
                    for pair in sorted_dist_vector:
                        f.write(" " + str(pair[0]) + " " + pair[1])
                    f.write("\n")