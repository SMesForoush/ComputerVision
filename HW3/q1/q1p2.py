import math

import numpy as np
import cv2
from matplotlib import pyplot as plt

vx = [9364, 2596, 1]
vy = [-2.6062e+04, 3946, 1]
vz = [-4.08478635e+03, -1.49838415e+05, 1]
"""
[2.68308212e+03 9.94976680e+03 1.00000000e+00]
[ 3.94677895e+03 -2.60624791e+04  1.00000000e+00]
[-1.49838415e+05 -4.08478635e+03  1.00000000e+00]
"""


def compute_rotation_matrix(x, y, z):
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]], dtype=np.float32)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]], dtype=np.float32)
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]], dtype=np.float32)
    R = np.matmul(Rz, np.matmul(Ry, Rx))
    return R


def compute_homography_matrix(camera_matrix, x, y, z):
    rotation_matrix = compute_rotation_matrix(x, y, z)
    print(rotation_matrix)
    homography = np.matmul(camera_matrix, np.matmul(rotation_matrix, np.linalg.inv(camera_matrix)))
    return homography


def compute_k():
    img = cv2.imread("../inputs/vns.jpg")
    a1 = vx[0]
    b1 = vx[1]

    a2 = vy[0]
    b2 = vy[1]

    a3 = vz[0]
    b3 = vz[1]

    matrix = [[a1 - a3, b1 - b3], [a2 - a3, b2 - b3]]
    b = [a2 * (a1 - a3) + b2 * (b1 - b3), a1 * (a2 - a3) + b1 * (b2 - b3)]
    x = np.linalg.solve(matrix, b)
    x = x.astype(np.int)
    plt.imshow(img)
    plt.scatter([x[0]], [x[1]])
    plt.axis("off")
    # plt.show()
    print(x)
    f2 = -x[0]**2-x[1]**2+(a1+a2)*x[0]+(b1+b1)*x[1]-(a1*a2+b1*b2)
    print(math.sqrt(f2))
    f = math.sqrt(f2)
    plt.title(f"f = {f}")
    plt.savefig("res03.jpg")
    # plt.show()
    K = np.array([[f, 0, x[0]], [0, f, x[1]], [0, 0, 1]])
    h = np.array([-3.50691684e-02,  -9.99384888e-01,  0])
    z = np.array([1, 0, 0])
    theta = np.dot(h, z)/np.sqrt(np.sum(h**2)+np.sum(z**2))
    print(theta)
    homography = compute_homography_matrix(K, 0, 0, theta)
    print(homography)
    new_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
    plt.imshow(new_img)
    plt.show()



compute_k()

"""
normalized h :  [-2.58632279e-02 -9.99665491e-01  2.48509382e+03]
"""
