import sys
import os
import glob

import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pickle

import feature_extractor as fe

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.stderr.write("Wrong usage: " + sys.argv[0] + " <PATH_TO_DATASET> <K-MEANS> <STEP_SIZE>\n")
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
    # Extract All Local Feature Descriptors
    if os.path.exists(os.path.join(CACHE, descriptor_filename + ".npy")):
        descriptors = np.load(os.path.join(CACHE, descriptor_filename + ".npy"))
    else:
        descriptors = np.zeros((0,128))

        for img in TRAIN_IMAGE_PATHS:
            des = fe.extract_feature(sift, img, step_size)

            try:
                descriptors = np.concatenate((descriptors, des), axis=0)
            except ValueError:
                #print("LocalFeatureDescriptor::ValueError:", img)
                pass
        
        np.save(os.path.join(CACHE, descriptor_filename), descriptors)
    
    print(descriptor_filename)


    kmeans_model_path = "kmeans" + str(k_means) + "_" + descriptor_filename 
    # K-Means Clustering
    if os.path.exists(os.path.join(CACHE, kmeans_model_path + ".model")):
        kmeans = pickle.load(open(os.path.join(CACHE, kmeans_model_path + ".model"), 'rb'))
    else:
        kmeans = MiniBatchKMeans(n_clusters=k_means, random_state=0).fit(descriptors)
        pickle.dump(kmeans, open(os.path.join(CACHE, kmeans_model_path + ".model"), 'wb'))

    print(kmeans_model_path)


    train_bow_path = "trainbow_" + kmeans_model_path
    # Extract Training Data Bag of Words
    if os.path.exists(os.path.join(CACHE, train_bow_path + ".npy")):
        train_histograms = np.load(os.path.join(CACHE, train_bow_path + ".npy"))
    else:
        train_histograms = np.empty((0, k_means))

        for img in TRAIN_IMAGE_PATHS:
            des = fe.extract_feature(sift, img, step_size)
            try:
                hist, _ = np.histogram(kmeans.predict(des), bins=range(k_means+1), normed=True)
            except ValueError:
                #print("TrainBoW::ValueError:", img)
                hist = np.zeros(k_means)
            train_histograms = np.concatenate((train_histograms, np.array([hist])), axis=0)
        
        np.save(os.path.join(CACHE, train_bow_path), train_histograms)

    print(train_bow_path)



    val_bow_path = "valbow_" + kmeans_model_path
    # Extract Validation Data Bag of Words
    if os.path.exists(os.path.join(CACHE, val_bow_path + ".npy")):
        val_histograms = np.load(os.path.join(CACHE, val_bow_path + ".npy"))
    else:
        val_histograms = np.empty((0, k_means))

        for img in VAL_IMAGE_PATHS:
            des = fe.extract_feature(sift, img, step_size)
            try:
                hist, _ = np.histogram(kmeans.predict(des), bins=range(k_means+1), normed=True)
            except ValueError:
                #print("ValBoW::ValueError:", img)
                hist = np.zeros(k_means)
            val_histograms = np.concatenate((val_histograms, np.array([hist])), axis=0)
        
        np.save(os.path.join(CACHE, val_bow_path), val_histograms)

    print(val_bow_path)


