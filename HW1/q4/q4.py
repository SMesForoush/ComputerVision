import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import lstsq
from icecream import ic
import math
output_path = 'outputs/'



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


# def compute_homography(points1, points2):
#     h = cv2.getPerspectiveTransform(points1, points2)
#     return h


def compute_difference(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def compute_inliers(h, points1, points2, threshold):
    inliers_p1, inliers_p2 = [], []
    mask = np.zeros(points1.shape[0])
    for i, p in enumerate(points1):
        xs = np.array([[points1[i][0, 0], points1[i][0, 1], 1]]).T
        predictions = np.matmul(h, xs)
        x_pred = predictions[0, 0]/predictions[2, 0]
        y_pred = predictions[1, 0]/predictions[2, 0]
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
    arr1 = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    arr2 = [x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp]
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
    # print("src points : ", points2)

    h, m = cv2.findHomography(srcPoints=np.array(points2), dstPoints=np.array(points1))
    # matrix = []
    # result = []
    # for j in range(len(points1)):
    #     p1 = points1[j]
    #     p2 = points2[j]
    #     arr1, arr2 = compute_points_array(p1, p2)
    #     matrix.append(arr1)
    #     matrix.append(arr2)
    #     result.append(0)
    #     result.append(0)
    # result = np.array(result)
    # matrix = np.array(matrix)
    # print("result : ", result.shape)
    # print("matrix : ", matrix.shape)
    # h, res, rnk, s = lstsq(matrix, result)
    # U, s, V = np.linalg.svd(matrix, full_matrices=True)
    # print(V.shape)
    # h = V[-1, :] / V[-1, -1]
    return h


def draw_match(base_image, position1, position2, color, r=5, thickness=5):
    cv2.circle(base_image, position1, r, color, thickness)
    cv2.circle(base_image, position2, r, color, thickness)
    cv2.line(base_image, position1, position2, color, thickness)


def draw_inlier_outlier(draw_img, based_shape, matches, masks, img1_points, img2_points,
                        inlier_color=(255, 0, 0), outlier_color=(0, 255, 255)):
    for i, m in enumerate(matches):
        end1 = (round(img1_points[i][0, 0]), round(img1_points[i][0, 1]))
        end2 = (round(img2_points[i][0, 0] + based_shape[1]), round(img2_points[i][0, 1]))
        if masks[i]:
            draw_match(base_image=draw_img, position1=end1, position2=end2, color=inlier_color)
        # else:
        #     draw_match(base_image=draw_img, position1=end1, position2=end2, color=outlier_color)
    return draw_img


def merge_images(img1, img2):
    new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    new_img = np.zeros(new_shape, dtype=img1.dtype)
    new_img[:img1.shape[0], :img1.shape[1]] = img1
    new_img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    return new_img


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
    ic(best_inlier_num)
    ic(best_h)
    final_homography = compute_lstsqr_homography(best_result)

    return best_h, final_mask


if __name__ == "__main__":
    img1 = load_image("../inputs/im03.jpg")
    img2 = load_image("../inputs/im04.jpg")
    transforming_img = img2.copy()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    new_matches = []
    for (m, n) in matches:
        if m.distance < 0.7 * n.distance:
            new_matches.append(m)

    image_1_points = np.zeros((len(new_matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(new_matches), 1, 2), dtype=np.float32)

    for i in range(len(new_matches)):
        image_1_points[i] = kp1[new_matches[i].queryIdx].pt
        image_2_points[i] = kp2[new_matches[i].trainIdx].pt
    homography, mask = ransac(image_1_points, image_2_points, 5000)
    finlier_outlier_images = draw_inlier_outlier(merge_images(img1, img2), img1.shape, new_matches, mask,
                                                 image_1_points, image_2_points)
    plt.imshow(finlier_outlier_images)
    plt.show()
    # h_matrix = homography.reshape((3, 3))
    h_matrix = homography
    h_matrix = h_matrix
    # ic(h_matrix)
    # h_matrix = h_matrix/homography[8]
    # ic(h_matrix)
    #
    projected_im = cv2.warpPerspective(transforming_img, h_matrix, (7000, 7000))
    plt.imshow(projected_im)
    plt.show()

"""
[[-3.72054958e+00 -9.54013943e+00  1.88899338e+03]
 [-1.31066656e+00 -9.69767633e+00  7.38441546e+02]
 [-3.11190479e-03 -1.07859328e-02  1.00000000e+00]]
"""