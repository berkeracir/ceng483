import sys

import cv2
import numpy as np

import random

FIXED_STEP_SIZE = 4

def extract_feature(sift, img, step_size=0):
    if isinstance(img, str):
        img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if step_size == 0:
        _, des = sift.detectAndCompute(gray, None)
    elif step_size == -1:
        step_size = FIXED_STEP_SIZE
        kp = [cv2.KeyPoint(x, y, step_size)
                for y in random.sample(range(0, gray.shape[0]), step_size)
                for x in random.sample(range(0, gray.shape[1]), step_size)]
        _, des = sift.compute(gray, kp)
    else:
        kp = [cv2.KeyPoint(x, y, step_size)
                for y in range(0, gray.shape[0], step_size)
                for x in range(0, gray.shape[1], step_size)]
        _, des = sift.compute(gray, kp)
    
    return des


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Wrong usage: " + sys.argv[0] + " <PATH_TO_IMAGE>")
        sys.exit(1)
    
    # img = cv2.imread(sys.argv[1])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = extract_feature(sift, sys.argv[1], 3)

    # img = cv2.drawKeypoints(gray, kp,       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=img)
    # cv2.imshow('sift_keypoints.jpg', img)
    # cv2.waitKey()