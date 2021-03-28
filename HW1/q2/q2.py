import cv2
import os
import math
import numpy as np
from matplotlib import pyplot as plt
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
    cv2.imshow("logo", logo)
    w, h, c = logo.shape
    px = w/2
    py = h/2
    f = 500
    img_shape = (10*w, 10*h)
    n = np.array([[0, 0, -1]])
    C = np.array([[0, -40, 0]]).T
    theta = math.atan(40/25)

    k = np.array([[f, 0, img_shape[1]/2], [0, f, img_shape[0]/2], [0, 0, 1]])
    k2 = np.array([[f, 0, py], [0, f, px], [0, 0, 1]])
    cos = math.cos(theta)
    sin = math.sin(theta)
    Ry = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    R = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
    Rz = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    t = -1 * np.matmul(R, C)
    tn = np.matmul(t, n)/25
    matrix = R - tn
    H = np.matmul(k2, matrix)
    H = np.matmul(H, np.linalg.inv(k))
    H_inverse = np.linalg.inv(H)
    result = cv2.warpPerspective(logo, H_inverse, img_shape)
    cv2.imwrite(output_path.format("res12.jpg"), result)



