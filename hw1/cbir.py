import os
import sys
import math
import numpy as np
import cv2
import glob

import color_histogram as color_hist
import gradient_histogram as grad_hist
import grayscale_histogram as gray_hist

#DATASET_PATH = "dataset"
OUTPUT_PATH = "out"
CACHE_PATH = "cache"
CACHE = True

horizontal_kernel = np.array([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]])

vertical_kernel = np.array([[1, 2, 1], 
                            [0, 0, 0], 
                            [-1, -2, -1]])

def usage():
    sys.stderr.write("Usage: python3 " + sys.argv[0] + " feature level bin query_images.dat image_dataset.dat dataset_path\n" + 
                    "   Feature: gray color grad\n"
                    "   Level: [1-6]\n"
                    "   Bin: [1-256] for grayscale and color, [1-360] for gradient\n"
                    "   Query File: Path to query_images.dat\n"
                    "   Dataset File: Path to image_dataset.dat\n"
                    "   Dataset: Path to dataset\n")
    sys.exit(1)

def get_histogram(feature, img, level, bin):
    if feature == "gray":
        return gray_hist.grayscale_histogram(img, level, bin)
    elif feature == "color":
        return color_hist.color_histogram(img, level, bin)
    elif feature == "grad":
        return grad_hist.gradient_histogram(img, vertical_kernel, horizontal_kernel, level, bin)
    else:
        sys.stderr.write("Wrong feature: " + feature + " (gray, color, grad)\n")
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
        d = math.sqrt(sum(sum((hist1-hist2)**2)))
    else:
        d = math.sqrt(sum(sum(sum(sum((hist1-hist2)**2)))))
 
    euclidean_distance = math.sqrt(d)
    return euclidean_distance

if __name__ == "__main__":
    if len(sys.argv) != 7: usage()

    FEATURE = sys.argv[1]
    LEVEL = int(sys.argv[2])
    BIN = int(sys.argv[3])
    QUERY_FILE = sys.argv[4]
    DATASET_FILE = sys.argv[5]
    DATASET_PATH = sys.argv[6]

    print("Feature:", FEATURE, "Level:", LEVEL, "Bin:", BIN)
    print("Query File:", QUERY_FILE)
    print("Dataset File:", DATASET_FILE)
    print("Dataset:", DATASET_PATH)

    if not os.path.exists(DATASET_PATH):
        sys.stderr.write("Dataset folder does not exist \""
                        + DATASET_PATH + "\"\n")
        sys.exit(1)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    if not os.path.exists(CACHE_PATH):
        caching = True

    query_images = []
    dataset_images = []

    with open(QUERY_FILE, "r") as query_file:
        for query_line in query_file:
            img_name = query_line.strip("\r\n")
            query_images.append(img_name)

    with open(DATASET_FILE, "r") as dataset_file:
        for dataset_line in dataset_file:
            img_name = dataset_line.strip("\r\n")
            dataset_images.append(img_name)

    cache = {}

    if FEATURE == "grad":
        conf = FEATURE + "_l" + str(LEVEL) + "b" + str(BIN) + "k1"
    else:
        conf = FEATURE + "_l" + str(LEVEL) + "b" + str(BIN)

    if os.path.exists(os.path.join(CACHE_PATH, conf)):
        caching = True

    fname = conf + ".txt"

    with open(os.path.join(OUTPUT_PATH, fname), "w") as f:
        for query_image in query_images:
            print(query_image)
            dist_vector = []

            qimg_name = query_image.split(".")[0]

            if query_image in cache:
                qhist = cache[query_image]
            elif caching:
                try:
                    qhist = np.load(os.path.join(CACHE_PATH, conf,
                                                    qimg_name + ".npy"))
                    cache[query_image] = qhist
                except FileNotFoundError:
                    qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                    qhist = get_histogram(FEATURE, qimg, LEVEL, BIN)
                    #np.save(os.path.join(CACHE_PATH, conf, qimg_name), qhist)
                    cache[query_image] = qhist
            else:
                qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                qhist = get_histogram(FEATURE, qimg, LEVEL, BIN)
                cache[query_image] = qhist

            for image in dataset_images:
                img_name = image.split(".")[0]

                if image in cache:
                    hist = cache[image]
                elif caching:
                    try:
                        hist = np.load(os.path.join(CACHE_PATH, conf,
                                                        img_name + ".npy"))
                        cache[image] = hist
                    except FileNotFoundError:
                        img = cv2.imread(os.path.join(DATASET_PATH, image))
                        hist = get_histogram(FEATURE, img, LEVEL, BIN)
                        #np.save(os.path.join(CACHE_PATH, conf, img_name), hist)
                        cache[image] = hist
                else:
                    img = cv2.imread(os.path.join(DATASET_PATH, image))
                    hist = get_histogram(FEATURE, img, LEVEL, BIN)
                    cache[image] = hist

                dist = euclidean_distance(qhist, hist)
                dist_vector.append((dist, image))

            sorted_dist_vector = sorted(dist_vector, key=lambda x: x[0])

            f.write(query_image + ":")
            for pair in sorted_dist_vector:
                f.write(" " + str(pair[0]) + " " + pair[1])
            f.write("\n")

        print("Output is written into " + os.path.join(OUTPUT_PATH, fname) + "\n")

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

    """
    color_levels = [1]
    color_bin_conf = [64] # [64, 32, 16, 8, 4, 2]

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

                    if query_image in cache:
                        qhist = cache[query_image]
                    else:
                        try:
                            qhist = np.load(os.path.join(CACHE_PATH, conf,
                                                            qimg_name + ".npy"))
                            cache[query_image] = qhist
                        except FileNotFoundError:
                            qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                            qhist = color_hist.color_histogram(qimg, level, bin)
                            np.save(os.path.join(CACHE_PATH, conf, qimg_name), qhist)
                            cache[query_image] = qhist

                    for image in dataset_images:
                        img_name = image.split(".")[0]

                        if image in cache:
                            hist = cache[image]
                        else:
                            try:
                                hist = np.load(os.path.join(CACHE_PATH, conf,
                                                                img_name + ".npy"))
                                cache[image] = hist
                            except FileNotFoundError:
                                img = cv2.imread(os.path.join(DATASET_PATH, image))
                                hist = color_hist.color_histogram(img, level, bin)
                                np.save(os.path.join(CACHE_PATH, conf, img_name), hist)
                                cache[image] = hist

                        dist = euclidean_distance(qhist, hist)
                        dist_vector.append((dist, image))

                    sorted_dist_vector = sorted(dist_vector, key=lambda x: x[0])

                    f.write(query_image + ":")
                    for pair in sorted_dist_vector:
                        f.write(" " + str(pair[0]) + " " + pair[1])
                    f.write("\n")
    """

    """
    grad_levels = [1]
    grad_bin_conf = [360, 72, 36, 24, 12, 6, 4, 2, 1]
    grad_kernel_conf = [0, 1]

    for kernel in grad_kernel_conf:
        for level in grad_levels:
            for bin in grad_bin_conf:
                cache = {}

                conf = "grad_l" + str(level) + "b" + str(bin) + "k" + str(kernel)

                if not os.path.exists(os.path.join(CACHE_PATH, conf)):
                    os.path.makedirs(os.path.join(CACHE_PATH, conf))

                fname = conf + ".txt"

                with open(os.path.join(OUTPUT_PATH, fname), "w") as f:
                    for query_image in query_images:
                        print(query_image)
                        dist_vector = []

                        qimg_name = query_image.split(".")[0]

                        if query_image in cache:
                            qhist = cache[query_image]
                        else:
                            try:
                                qhist = np.load(os.path.join(CACHE_PATH, conf,
                                                                qimg_name + ".npy"))
                                cache[query_image] = qhist
                            except FileNotFoundError:
                                qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                                qhist = color_hist.color_histogram(qimg, level, bin) # TODO
                                np.save(os.path.join(CACHE_PATH, conf, qimg_name), qhist)
                                cache[query_image] = qhist

                        for image in dataset_images:
                            img_name = image.split(".")[0]

                            if image in cache:
                                hist = cache[image]
                            else:
                                try:
                                    hist = np.load(os.path.join(CACHE_PATH, conf,
                                                                    img_name + ".npy"))
                                    cache[image] = hist
                                except FileNotFoundError:
                                    img = cv2.imread(os.path.join(DATASET_PATH, image))
                                    hist = color_hist.color_histogram(img, level, bin)
                                    np.save(os.path.join(CACHE_PATH, conf, img_name), hist)
                                    cache[image] = hist

                            dist = euclidean_distance(qhist, hist)
                            dist_vector.append((dist, image))

                        sorted_dist_vector = sorted(dist_vector, key=lambda x: x[0])

                        f.write(query_image + ":")
                        for pair in sorted_dist_vector:
                        f.write(" " + str(pair[0]) + " " + pair[1])
                        f.write("\n")
    """

    """
    levels = [2, 3]
    bin = 16

    for level in levels:
        conf = "gray_l" + str(level) + "b" + str(bin)

        if not os.path.exists(os.path.join(CACHE_PATH, conf)):
            os.path.makedirs(os.path.join(CACHE_PATH, conf))

        fname = conf + ".txt"
        cache = {}

        with open(os.path.join(OUTPUT_PATH, fname), "w") as f:
            for query_image in query_images:
                print(query_image)
                dist_vector = []

                qimg_name = query_image.split(".")[0]

                if query_image in cache:
                    qhist = cache[query_image]
                else:
                    try:
                        qhist = np.load(os.path.join(CACHE_PATH, conf,
                                                        qimg_name + ".npy"))
                        cache[query_image] = qhist
                    except FileNotFoundError:
                        qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                        qhist = gray_hist.grayscale_histogram(qimg, level, bin)
                        np.save(os.path.join(CACHE_PATH, conf, qimg_name), qhist)
                        cache[query_image] = qhist

                for image in dataset_images:
                    img_name = image.split(".")[0]

                    if image in cache:
                        hist = cache[image]
                    else:
                        try:
                            hist = np.load(os.path.join(CACHE_PATH, conf,
                                                            img_name + ".npy"))
                            cache[image] = hist
                        except FileNotFoundError:
                            img = cv2.imread(os.path.join(DATASET_PATH, image))
                            hist = gray_hist.grayscale_histogram(img, level, bin)
                            np.save(os.path.join(CACHE_PATH, conf, img_name), hist)
                            cache[image] = hist

                    dist = euclidean_distance(qhist, hist)
                    dist_vector.append((dist, image))

                sorted_dist_vector = sorted(dist_vector, key=lambda x: x[0])

                f.write(query_image + ":")
                for pair in sorted_dist_vector:
                    f.write(" " + str(pair[0]) + " " + pair[1])
                f.write("\n")

    for level in levels:
        conf = "color_l" + str(level) + "b" + str(bin)

        if not os.path.exists(os.path.join(CACHE_PATH, conf)):
            os.path.makedirs(os.path.join(CACHE_PATH, conf))

        fname = conf + ".txt"
        cache = {}

        with open(os.path.join(OUTPUT_PATH, fname), "w") as f:
            for query_image in query_images:
                print(query_image)
                dist_vector = []

                qimg_name = query_image.split(".")[0]

                if query_image in cache:
                    qhist = cache[query_image]
                else:
                    try:
                        qhist = np.load(os.path.join(CACHE_PATH, conf,
                                                        qimg_name + ".npy"))
                        cache[query_image] = qhist
                    except FileNotFoundError:
                        qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                        qhist = color_hist.color_histogram(qimg, level, bin)
                        np.save(os.path.join(CACHE_PATH, conf, qimg_name), qhist)
                        cache[query_image] = qhist

                for image in dataset_images:
                    img_name = image.split(".")[0]

                    if image in cache:
                        hist = cache[image]
                    else:
                        try:
                            hist = np.load(os.path.join(CACHE_PATH, conf,
                                                            img_name + ".npy"))
                            cache[image] = hist
                        except FileNotFoundError:
                            img = cv2.imread(os.path.join(DATASET_PATH, image))
                            hist = color_hist.color_histogram(img, level, bin)
                            np.save(os.path.join(CACHE_PATH, conf, img_name), hist)
                            cache[image] = hist

                    dist = euclidean_distance(qhist, hist)
                    dist_vector.append((dist, image))

                sorted_dist_vector = sorted(dist_vector, key=lambda x: x[0])

                f.write(query_image + ":")
                for pair in sorted_dist_vector:
                    f.write(" " + str(pair[0]) + " " + pair[1])
                f.write("\n")

    bin = 360

    for level in levels:
        conf = "grad_l" + str(level) + "b" + str(bin) + "k1"

        if not os.path.exists(os.path.join(CACHE_PATH, conf)):
            os.path.makedirs(os.path.join(CACHE_PATH, conf))

        fname = conf + ".txt"
        cache = {}

        with open(os.path.join(OUTPUT_PATH, fname), "w") as f:
            for query_image in query_images:
                print(query_image)
                dist_vector = []

                qimg_name = query_image.split(".")[0]

                if query_image in cache:
                    qhist = cache[query_image]
                else:
                    try:
                        qhist = np.load(os.path.join(CACHE_PATH, conf,
                                                        qimg_name + ".npy"))
                        cache[query_image] = qhist
                    except FileNotFoundError:
                        print("ERR")
                        exit(1)
                        #qimg = cv2.imread(os.path.join(DATASET_PATH, query_image))
                        #qhist = color_hist.color_histogram(qimg, level, bin)
                        #np.save(os.path.join(CACHE_PATH, conf, qimg_name), qhist)
                        #cache[query_image] = qhist

                for image in dataset_images:
                    img_name = image.split(".")[0]

                    if image in cache:
                        hist = cache[image]
                    else:
                        try:
                            hist = np.load(os.path.join(CACHE_PATH, conf,
                                                            img_name + ".npy"))
                            cache[image] = hist
                        except FileNotFoundError:
                            print("ERR")
                            exit(1)
                            #img = cv2.imread(os.path.join(DATASET_PATH, image))
                            #hist = color_hist.color_histogram(img, level, bin)
                            #np.save(os.path.join(CACHE_PATH, conf, img_name), hist)
                            #cache[image] = hist

                    dist = euclidean_distance(qhist, hist)
                    dist_vector.append((dist, image))

                sorted_dist_vector = sorted(dist_vector, key=lambda x: x[0])

                f.write(query_image + ":")
                for pair in sorted_dist_vector:
                    f.write(" " + str(pair[0]) + " " + pair[1])
                f.write("\n")
    """