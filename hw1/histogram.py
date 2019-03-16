import sys
import numpy as np
import cv2
from PIL import Image

def usage():
    sys.stderr.write("Usage: python " + sys.argv[0] + " img.jpg\n")
    sys.exit(1)

def grayscale_histogram(image, bin):
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2: usage()

    img = cv2.imread(sys.argv[1])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("result", img)
    cv2.waitKey(0)