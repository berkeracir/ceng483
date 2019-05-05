import sys
import os
import glob
import matplotlib.pyplot as plt

import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pickle

from collections import Counter

import feature_extractor as fe

def predict(train_images, indices):
    classes = []
    for index in indices:
        classes.append(train_images[index].split("/")[-2])
    data = Counter(classes).most_common() 
    predicted_classes = [name for name, count in data if count == data[0][1]]

    for class_name in classes:
        if class_name in predicted_classes:
            return class_name

if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.stderr.write("Wrong usage: " + sys.argv[0] + " <PATH_TO_DATASET> <PATH_TO_TEST_IMAGES> <K-MEANS> <STEP_SIZE> <kNN>\n")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        sys.stderr.write("Wrong path to dataset: " + sys.argv[1]
             + " (Dataset does NOT exist!)\n")
        sys.exit(1)

    DATASET = sys.argv[1]
    TEST_IMAGES = sys.argv[2]
    TRAIN_DATASET = os.path.join(DATASET, "train")
    VAL_DATASET = os.path.join(DATASET, "validation")
    CLASS_NAMES = []
    TRAIN_IMAGE_PATHS = []
    VAL_IMAGE_PATHS = []
    TEST_IMAGE_PATHS = []
    CACHE = "cache"
    k_means = int(sys.argv[3])
    step_size = int(sys.argv[4])
    knn = int(sys.argv[5])

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

    for img_path in glob.glob(os.path.join(TEST_IMAGES,  "*")):
        TEST_IMAGE_PATHS.append(img_path)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.014, edgeThreshold=10, sigma=0.27)
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

    # K-Nearest Neighbors
    for test_img in TEST_IMAGE_PATHS:
        des = fe.extract_feature(sift, test_img, step_size)
        try:
            hist, _ = np.histogram(kmeans.predict(des), bins=range(k_means+1), normed=True)
        except ValueError:
            des = fe.extract_feature(sift, test_img, 16)
            hist, _ = np.histogram(kmeans.predict(des), bins=range(k_means+1), normed=True)
        distances = np.square(train_histograms-hist).sum(axis=1)
        idx = np.argpartition(distances, knn)
        prediction = predict(TRAIN_IMAGE_PATHS, idx[:knn])
        print(test_img.split("/")[-1] + ":", prediction)

