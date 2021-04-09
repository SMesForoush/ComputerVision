import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import lstsq
from icecream import ic
import math
from HW1.q3 import q31


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


def compute_difference(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_inliers(h, points1, points2, threshold):
    inliers_p1, inliers_p2 = [], []
    mask = np.zeros(points1.shape[0])
    for i, p in enumerate(points1):
        xs = np.array([[points1[i][0, 0], points1[i][0, 1], 1]]).T
        predictions = np.matmul(h, xs)
        x_pred = predictions[0, 0] / predictions[2, 0]
        y_pred = predictions[1, 0] / predictions[2, 0]
        diff = compute_difference(x_pred, y_pred, points2[i][0, 0], points2[i][0, 1])
        # print(f"diff {i} : ", diff)
        if diff < threshold:
            inliers_p1.append([[points1[i][0, 0], points1[i][0, 1]]])
            inliers_p2.append([[points2[i][0, 0], points2[i][0, 1]]])
            mask[i] = 1

    return inliers_p1, inliers_p2, mask


def compute_points_array(point1, point2):
    xp = point1[0, 0]
    x = point2[0, 0]
    yp = point1[0, 1]
    y = point2[0, 1]
    arr1 = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]
    arr2 = [x, y, 1, 0, 0, 0, -x * xp, -y * xp, -xp]
    return arr1, arr2


def compute_points_matrix(points1, points2):
    matrix = []
    for j in range(len(points1)):
        p1 = points1[j]
        p2 = points2[j]
        arr1, arr2 = compute_points_array(p1, p2)
        matrix.append(arr1)
        matrix.append(arr2)
    matrix = np.array(matrix)
    return matrix


def compute_homography(points1, points2):
    A = compute_points_matrix(points1, points2)
    U, s, V = np.linalg.svd(A)
    h = V.T[:, -1]
    return h.reshape((3, 3))


def compute_lstsqr_homography(inliers):
    points1, points2 = inliers
    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)

    # h, m = cv2.findHomography(srcPoints=np.array(points2), dstPoints=np.array(points1))
    matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 1]]
    # matrix = []
    result = [1]
    for j in range(len(points1)):
        p1 = points1[j]
        p2 = points2[j]
        arr1, arr2 = compute_points_array(p1, p2)
        matrix.append(arr1)
        matrix.append(arr2)
        result.append(0)
        result.append(0)
    result = np.array(result, dtype=np.float32)
    matrix = np.array(matrix, dtype=np.float32)
    lst_h, res, rnk, s = lstsq(matrix, result)
    lst_h = lst_h.reshape((3, 3))
    lst_h = lst_h / lst_h[2, 2]
    svd_h = compute_homography(points1, points2)
    svd_h = svd_h/svd_h[2, 2]
    return svd_h


def ransac(points1, points2, N):
    list_range = range(len(points1))
    sample_num = 4
    best_result = None
    best_inlier_num = -1
    threshold = 100
    final_mask = None
    best_h = None
    for i in range(N):
        sample_indexes = random.sample(list_range, sample_num)
        sample_points1 = points1[sample_indexes]
        sample_points2 = points2[sample_indexes]
        h = compute_homography(sample_points1, sample_points2)
        h = h / h[2, 2]
        inliers1, inliers2, mask = compute_inliers(h, points2, points1, threshold)
        inliers_num = len(inliers1)
        if inliers_num > best_inlier_num:
            best_inlier_num = inliers_num
            best_result = (inliers1, inliers2)
            final_mask = mask
            best_h = h
    final_homography = compute_lstsqr_homography(best_result)
    final_homography = final_homography / final_homography[2, 2]
    return best_h, final_mask


if __name__ == "__main__":
    img1 = load_image("../inputs/im03.jpg")
    img2 = load_image("../inputs/im04.jpg")
    new_matches, match_points, key_points = q31.compute_matching_points(img1, img2)
    image_1_points = match_points[0]
    image_2_points = match_points[1]
    homography, mask = ransac(image_1_points, image_2_points, 7000)
    # finlier_outlier_images = q31.draw_inlier_outlier(q31.merge_images(img1, img2), img1.shape, new_matches, mask,
    #                                                  image_1_points, image_2_points)
    # plt.imshow(finlier_outlier_images)
    # plt.show()
    h_matrix = homography
    transforming_img = img2.copy()
    # h_matrix = np.array([[1.7798548e+00, 6.6127861e-01, 1.8474008e+02],
    #                      [1.1618153e+00, 1.9653734e+00, -2.3575032e+02],
    #                      [1.1933770e-03, 9.3739229e-04, 1.0000000e+00]], dtype=np.float32)
    projected_im = cv2.warpPerspective(transforming_img, h_matrix, (1000, 1000))
    cv2.imwrite("res20.jpg", projected_im)

"""
ic| best_h: array([[ 3.52557214e+00,  2.80726590e-01, -2.26172914e+03],
                   [ 1.00424726e-01,  2.26160782e+00, -1.15416081e+03],
                   [ 1.20562000e-04, -1.88338772e-04,  1.00000000e+00]])
ic| h: array([ 1.2596006e-06,  4.6798590e-07,  1.3074029e-04,  8.2221499e-07,
               1.3908918e-06, -1.6684015e-04,  8.4455110e-10,  6.6339112e-10,
               7.0769852e-07], dtype=float32)
ic| h_matrix: array([[ 1.7798548e+00,  6.6127861e-01,  1.8474008e+02],
                     [ 1.1618153e+00,  1.9653734e+00, -2.3575032e+02],
                     [ 1.1933770e-03,  9.3739229e-04,  1.0000000e+00]], dtype=float32)

"""


"""
svd :  [[ 3.9264586e-02, -3.7883878e-02,  6.5775244e+02],
                             [-1.6091885e-01,  3.5049862e-01,  5.3717590e+02],
                             [-2.0622200e-04,  2.1815495e-05,  1.0000000e+00]]
                             
                             
lstsqrs : [[ 1.7798548e+00,  6.6127861e-01,  1.8474008e+02],
                             [ 1.1618153e+00,  1.9653734e+00, -2.3575032e+02],
                             [ 1.1933770e-03,  9.3739229e-04,  1.0000000e+00]]

"""