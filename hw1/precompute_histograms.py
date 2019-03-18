import os
import sys
import glob
import cv2
import numpy as np
import grayscale_histogram as gh
import color_histogram as ch

DATA_PATH = "data"
DATASET_PATH = "dataset"

GRAYSCALE = "grayscale"
COLOR = "color"
GRADIENT = "gradient"

QUANTIZATION_LEVELS = [256, 128, 64, 32, 16] # Possible Bins: [8, 4, 2, 1]
GRID_LEVELS = [1, 2, 3] # Possible Levels: [1, 6]

def precompute_grayscale_histograms(image_path):
    image_name = image_path.split("/")[-1]
    folder_name = image_name.split(".")[0]

    if not os.path.exists(os.path.join(DATA_PATH, folder_name, GRAYSCALE)):
        os.makedirs(os.path.join(DATA_PATH, folder_name, GRAYSCALE))

    for level in GRID_LEVELS:
        for bin in QUANTIZATION_LEVELS:
            image = cv2.imread(image_path)
            ghist = gh.grayscale_histogram(image, level, bin)
            fname = "l" + str(level) + "b" + str(bin)
            fpath = os.path.join(DATA_PATH, folder_name, GRAYSCALE, fname)
            np.save(fpath, ghist)

def precompute_3d_color_histograms(image_path):
    image_name = image_path.split("/")[-1]
    folder_name = image_name.split(".")[0]

    if not os.path.exists(os.path.join(DATA_PATH, folder_name, COLOR)):
        os.makedirs(os.path.join(DATA_PATH, folder_name, COLOR))

    for level in GRID_LEVELS:
        for bin in QUANTIZATION_LEVELS:
            image = cv2.imread(image_path)
            chist = ch.color_histogram(image, level, bin)
            fname = "l" + str(level) + "b" + str(bin)
            fpath = os.path.join(DATA_PATH, folder_name, COLOR, fname)
            np.save(fpath, chist)

#def precompute_gradient_histograms(image_path):

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        sys.stderr.write("Dataset folder does not exist \""
                        + DATASET_PATH + "\"\n")
        sys.exit(1)

    for img_path in glob.glob(os.path.join(DATASET_PATH, "*.jpg")):
        #precompute_grayscale_histograms(img_path)
        precompute_3d_color_histograms(img_path)
