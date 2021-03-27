import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from skimage.feature import corner_peaks
output_path = 'outputs/{}'


def img_resize(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output


def load_image(image_address):
    img = cv2.imread(image_address)
    # img = img_resize(img, 50)
    return img


if __name__ == '__main__':
    logo = load_image("../inputs/logo.png")
    px, py, c = logo.shape
    px = px/2
    py = py/2
    fx = 500
    fy = 500
    d = 25
    z = 40
    cv2.imshow("logo", logo)
    cv2.waitKey(0)
    k = [[fx, 0, px], [0, fy, py], [0, 0, 1]]
    k2 = [[fx, 0, 10*px], [0, fy, 10*py], [0, 0, 1]]
