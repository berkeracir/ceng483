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

import confusion_matrix as cm

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
        sys.stderr.write("Wrong usage: " + sys.argv[0] + " <PATH_TO_DATASET> <K-MEANS> <STEP_SIZE> <kNN>\n")
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
    k_means = int(sys.argv[2])
    step_size = int(sys.argv[3])
    knn = int(sys.argv[4])

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


    val_bow_path = "valbow_" + kmeans_model_path
    # Get Validation Data Bag of Words
    if os.path.exists(os.path.join(CACHE, val_bow_path + ".npy")):
        val_histograms = np.load(os.path.join(CACHE, val_bow_path + ".npy"))
    else:
        sys.stderr.write("Training Data BoW not found at " 
                    + os.path.join(CACHE, val_bow_path + ".npy") + "\n")
        sys.exit(1)

    true_classes = []
    predicted_classes = []
    # K-Nearest Neighbors
    if k_means > 64:
        index = 0
        mAP = 0
        for val_img in VAL_IMAGE_PATHS:
            distances = np.square(train_histograms-val_histograms[index]).sum(axis=1)
            idx = np.argpartition(distances, knn)
            correct_class = val_img.split("/")[-2]
            prediction = predict(TRAIN_IMAGE_PATHS, idx[:knn])
            if prediction == correct_class:
                mAP += 1
            index += 1
            true_classes.append(correct_class)
            predicted_classes.append(prediction)
    else:
        L2_distances = np.square(train_histograms[:, None] - val_histograms).sum(axis=2).T
        index = 0
        mAP = 0
        for val_img in VAL_IMAGE_PATHS:
            distances = L2_distances[index]
            idx = np.argpartition(distances, knn)
            correct_class = val_img.split("/")[-2]
            prediction = predict(TRAIN_IMAGE_PATHS, idx[:knn])
            if prediction == correct_class:
                mAP += 1
            index += 1
            true_classes.append(correct_class)
            predicted_classes.append(prediction)

    print("mAP:", mAP/len(VAL_IMAGE_PATHS), "\n")
    
    # Confusion Matrix
    confusion_matrix_title = "Accuraccy=" + str(mAP/len(VAL_IMAGE_PATHS)) + " (k-Means:" + str(k_means) + " StepSize:" + str(step_size) + " k-NN:" + str(knn) + ")"
    cm.plot_confusion_matrix(true_classes, predicted_classes, CLASS_NAMES, title=confusion_matrix_title)
    plt.show()
