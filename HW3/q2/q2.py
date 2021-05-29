import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
INPUT_ADDRESS = "../inputs/{}"


def merge_images(img1, img2):
    new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    new_img = np.zeros(new_shape, dtype=img1.dtype)
    new_img[:img1.shape[0], :img1.shape[1]] = img1
    new_img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    return new_img


def draw_match(base_image, position1, position2, color, r=5, thickness=5):
    cv2.circle(base_image, position1, r, color, thickness)
    cv2.circle(base_image, position2, r, color, thickness)
    # cv2.line(base_image, position1, position2, color, thickness)


def draw_inlier_outlier(draw_img, based_shape, masks, img1_points, img2_points,
                        inlier_color=(0, 255, 0), outlier_color=(0, 0, 255)):
    for i, imi in enumerate(img1_points):
        end1 = (round(img1_points[i][0, 0]), round(img1_points[i][0, 1]))
        end2 = (round(img2_points[i][0, 0] + based_shape[1]), round(img2_points[i][0, 1]))
        if masks[i]:
            draw_match(base_image=draw_img, position1=end1, position2=end2, color=inlier_color)
        else:
            draw_match(base_image=draw_img, position1=end1, position2=end2, color=outlier_color)
    return draw_img


def compute_corresponding_points(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    new_matches = []
    for (m, n) in matches:
        if m.distance < 0.8 * n.distance:
            new_matches.append(m)

    image_1_points = np.zeros((len(new_matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(new_matches), 1, 2), dtype=np.float32)

    for i in range(len(new_matches)):
        image_1_points[i] = kp1[new_matches[i].queryIdx].pt
        image_2_points[i] = kp2[new_matches[i].trainIdx].pt
    return image_1_points, image_2_points


def epipoleSVD(M):
    V = cv2.SVDecomp(M)[2]
    return V[-1]/V[-1,-1]


def compute_y(l, x):
    y = (-l[2]-l[0]*x)/l[1]
    return int(y[0])


def compute_x(l, y):
    x = (-l[2]-l[1]*y)/l[0]
    return x[0]


def create_random_color():
    r = random.random()*255
    b = random.random()*255
    g = random.random()*255
    color = (r, g, b)
    print("random color : ", color)
    return color


def create_epipolar_line(im2, l_prim, point, color):
    w, h, c = im2.shape
    y1 = compute_y(l_prim, 0)
    print("y1 : ", y1)
    y2 = compute_y(l_prim, h)
    cv2.line(im2, (0, y1), (h, y2), color, 10)
    cv2.circle(im2, (point[0], point[1]), 15, color, -1)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def compute_fundamental_matrix(im1, im2):
    p1, p2 = compute_corresponding_points(im1, im2)
    F, mask = cv2.findFundamentalMat(np.int32(p1), np.int32(p2), cv2.FM_RANSAC, 0.2, 0.9)
    # print("sahel : ", F)
    res = draw_inlier_outlier(merge_images(im1, im2), im1.shape, mask, p1, p2)
    cv2.imwrite("res05.jpg", res)
    e1 = epipoleSVD(F)
    e2 = epipoleSVD(F.T)
    # print(e1[0], e1[1])
    # print(e2[0], e2[1])
    # print(e2, e1)
    inlier_points1 = p1[mask == 1]
    inlier_points2 = p2[mask == 1]
    epi_line_img1 = im1.copy()
    epi_line_img2 = im2.copy()
    # print(inlier_points1)
    sample_indexes = random.sample(range(len(inlier_points1)), 10)
    for i in sample_indexes:
        point1 = inlier_points1[i]
        point2 = inlier_points2[i]
        x_array = np.array([[point1[0]], [point1[1]], [1]])
        x2_array = np.array([[point2[0]], [point2[1]], [1]])
        l_prim = np.matmul(F, x_array)
        l = np.matmul(F.T, x2_array)
        color = create_random_color()
        create_epipolar_line(epi_line_img2, l_prim, point2, color)
        create_epipolar_line(epi_line_img1, l, point1, color)

        print("l : ", l_prim)
    # cv2.circle(im2, e2, 20, (255, 0, 0), 20)
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(epi_line_img1)
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(epi_line_img2)
    plt.savefig("res08.jpg")
    plt.clf()
    plt.axis("off")
    plt.imshow(im1)
    plt.scatter([e1[0]], [e1[1]], s=20)
    plt.savefig("res06.jpg")
    plt.clf()
    plt.axis("off")
    plt.imshow(im2)
    plt.scatter([e2[0]], [e2[1]], s=20)
    plt.savefig("res07.jpg")
    # print(h)
    # plt.imshow(im1)
    # plt.show()


def q2():
    im1 = cv2.imread(INPUT_ADDRESS.format("01.JPG"))
    im2 = cv2.imread(INPUT_ADDRESS.format("02.JPG"))
    compute_fundamental_matrix(im1, im2)


q2()