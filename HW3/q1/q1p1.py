import json
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
INPUT_PATH = "../inputs/{}"


def compute_hough_lines(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(gray_img, 150, 200, None, 3)
    vert_lines = cv2.HoughLines(edges, 1, np.pi / 160, 470, min_theta=-0.15, max_theta=0)
    x_line = cv2.HoughLines(edges, 1, np.pi / 160, 390, min_theta=1.65, max_theta=1.75)
    y_line = cv2.HoughLines(edges, 1, np.pi / 160, 390, min_theta=1.4, max_theta=1.57)
    vx = compute_line_intersection(x_line)
    vx = vx/vx[2]
    vy = compute_line_intersection(y_line)
    vy = vy/vy[2]
    res1_img = img.copy()
    cv2.line(res1_img, (int(vx[0]), int(vx[1])), (int(vy[0]), int(vy[1])), (0, 0, 255), 5)
    cv2.imwrite("res01.jpg", res1_img)
    h = np.cross(vx, vy)
    print("h : ", h)
    r = math.sqrt(h[0]**2+h[1]**2)
    normalized_h = h/r
    print("normalized h : ", normalized_h)
    xax = [vx[0], vy[0]]
    yax = [vx[1], vy[1]]
    vz = compute_line_intersection(vert_lines)
    vz = vz/vz[2]
    plt.imshow(img)
    plt.plot(xax, yax, 'b-.')
    plt.scatter([[vx[0], vy[0], vz[0]]], [[[vx[1], vy[1], vz[1]]]])
    plt.savefig("res02.jpg")


def compute_line_intersection(lines):
    matrix = []
    for i in range(len(lines)):
        line = lines[i]
        r1, th1 = line[0]
        matrix.append([math.cos(th1), math.sin(th1), -r1])
    U, s, V = np.linalg.svd(matrix)
    result = V.T[:, -1]
    print(result/result[2])
    return result



img = cv2.imread(INPUT_PATH.format("vns.jpg"))
compute_hough_lines(img)
