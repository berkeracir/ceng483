import sys
import os
import glob

import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pickle

from collections import Counter

def predict(img_name, train_images, indices):
    correct_class = img_name.split("/")[-2]
    classes = []
    for index in indices:
        classes.append(train_images[index].split("/")[-2])
    data = Counter(classes)
    correct_class_count = data.get(correct_class)
    predicted_class_count = data.most_common(1)[0][1]
    if correct_class_count is None or correct_class_count < predicted_class_count:
        return 0
    else:
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Wrong usage: " + sys.argv[0] + " <PATH_TO_DATASET>\n")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        sys.stderr.write("Wrong path to dataset: " + sys.argv[1]
             + " (Dataset does NOT exist!)\n")
        sys.exit(1)

    DATASET = sys.argv[1]
    TRAIN_DATASET = os.path.join(DATASET, "train")
    VAL_DATASET = os.path.join(DATASET, "validation")
    CLASS_NAMES = []
    TRAIN_IMAGE_PATHS = []
    VAL_IMAGE_PATHS = []
    CACHE = "cache"
    k_means = 128
    step_size = 0
    knn = 1

    if not os.path.exists(CACHE):
        os.makedirs(CACHE)

    for path in glob.glob(os.path.join(TRAIN_DATASET, "*")):
        CLASS_NAMES.append(os.path.basename(path))

    for path in [os.path.join(TRAIN_DATASET, class_name, "*") for class_name in CLASS_NAMES]:
        for img_path in glob.glob(path):
            TRAIN_IMAGE_PATHS.append(img_path)

    for path in [os.path.join(VAL_DATASET, class_name, "*") for class_name in CLASS_NAMES]:
        for img_path in glob.glob(path):
            VAL_IMAGE_PATHS.append(img_path)
    
    
    sift = cv2.xfeatures2d.SIFT_create()
    descriptor_filename = "descriptor" + str(step_size)
    # Get All Local Feature Descriptors
    if os.path.exists(os.path.join(CACHE, descriptor_filename + ".npy")):
        descriptors = np.load(os.path.join(CACHE, descriptor_filename + ".npy"))
    else:
        sys.stderr.write("Local Descriptors not found at " 
                    + os.path.join(CACHE, descriptor_filename + ".npy") + "\n")
        sys.exit(1)


    kmeans_model_path = "kmeans" + str(k_means) + "_" + descriptor_filename 
    # K-Means Clustering
    if os.path.exists(os.path.join(CACHE, kmeans_model_path + ".model")):
        kmeans = pickle.load(open(os.path.join(CACHE, kmeans_model_path + ".model"), 'rb'))
    else:
        sys.stderr.write("K-Means Model not found at " 
                    + os.path.join(CACHE, kmeans_model_path + ".model") + "\n")
        sys.exit(1)


    train_bow_path = "trainbow_" + kmeans_model_path
    # Get Training Data Bag of Words
    if os.path.exists(os.path.join(CACHE, train_bow_path + ".npy")):
        train_histograms = np.load(os.path.join(CACHE, train_bow_path + ".npy"))
    else:
        sys.stderr.write("Training Data BoW not found at " 
                    + os.path.join(CACHE, train_bow_path + ".npy") + "\n")
        sys.exit(1)


    val_bow_path = "valbow_" + kmeans_model_path
    # Get Validation Data Bag of Words
    if os.path.exists(os.path.join(CACHE, val_bow_path + ".npy")):
        val_histograms = np.load(os.path.join(CACHE, val_bow_path + ".npy"))
    else:
        sys.stderr.write("Training Data BoW not found at " 
                    + os.path.join(CACHE, val_bow_path + ".npy") + "\n")
        sys.exit(1)

    # K-Nearest Neighbors
    L2_distances = np.square(train_histograms[:, None] - val_histograms).sum(axis=2).T
    index = 0
    mAP = 0
    for val_img in VAL_IMAGE_PATHS:
        distances = L2_distances[index]
        idx = np.argpartition(distances, knn)
        mAP += predict(val_img, TRAIN_IMAGE_PATHS, idx[:knn])
        index += 1

    print("mAP:", mAP/len(VAL_IMAGE_PATHS))
