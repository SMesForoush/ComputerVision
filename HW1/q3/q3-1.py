import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
output_path = 'outputs/knn-{}'


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
        else:
            draw_match(base_image=draw_img, position1=end1, position2=end2, color=outlier_color)
    return draw_img


def draw_correspondents(new_img, based_shape, img1_points, img2_points, matches, color=None):
    r = 5
    thickness = 5
    img_with_line = new_img.copy()
    for i, m in enumerate(matches):
        end1 = (round(img1_points[i][0, 0]), round(img1_points[i][0, 1]))
        end2 = (round(img2_points[i][0, 0] + based_shape[1]), round(img2_points[i][0, 1]))
        cv2.circle(new_img, end1, r, color, thickness)
        cv2.circle(new_img, end2, r, color, thickness)
        draw_match(img_with_line, end1, end2, color)
    return new_img, img_with_line


def merge_images(img1, img2):
    new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    new_img = np.zeros(new_shape, dtype=img1.dtype)
    new_img[:img1.shape[0], :img1.shape[1]] = img1
    new_img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    return new_img


if __name__ == '__main__':
    img1 = load_image("../inputs/im03.jpg")
    img2 = load_image("../inputs/im04.jpg")
    transforming_img = img2.copy()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    res = cv2.drawMatches(img1, kp1, img2, kp2, [], img2, flags=0, singlePointColor=(0, 255, 0))
    cv2.imwrite(output_path.format("res13_corners.jpg"), res)
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)

    matches = bf.knnMatch(des1, des2, k=2)
    new_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            new_matches.append(m)
    # best_twenty_matches = sorted(new_matches, key=lambda x: x.distance)[:40]
    best_twenty_matches = random.sample(new_matches, 20)
    best_twenty_images = cv2.drawMatches(img1, kp1, img2, kp2, best_twenty_matches, img2, flags=2, matchColor=(255, 0, 0))
    cv2.imwrite(output_path.format("res16-1.jpg"), best_twenty_images)

    image_1_points = np.zeros((len(new_matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(new_matches), 1, 2), dtype=np.float32)

    for i in range(len(new_matches)):
        image_1_points[i] = kp1[new_matches[i].queryIdx].pt
        image_2_points[i] = kp2[new_matches[i].trainIdx].pt

    # draw_params = dict(matchColor=(0, 0, 0),
    #                    singlePointColor=(0, 255, 0),
    #                    flags=2)
    img4, with_line = draw_correspondents(res, img1.shape, image_1_points, image_2_points, new_matches, color=(0, 0, 255))
    cv2.imwrite(output_path.format("res14_correspondences.jpg"), img4)
    cv2.imwrite(output_path.format("res15_match.jpg"), with_line)

    homography, mask = cv2.findHomography(image_2_points, image_1_points, cv2.RANSAC,
                                          maxIters=2000)
    inliersMask = mask.ravel().tolist()

    inlier_outlier_images = draw_inlier_outlier(merge_images(img1, img2), img1.shape, new_matches, inliersMask,
                                                image_1_points, image_2_points)
    cv2.imwrite(output_path.format("res17.jpg"), inlier_outlier_images)
    print(homography)
    projected_img = cv2.warpPerspective(transforming_img, homography, (10000, 10000))
    cv2.imwrite(output_path.format("res19.jpg"), projected_img)

"""
[[ 3.59177280e+00  3.07034125e-01 -2.35543670e+03]
 [ 6.05885788e-02  2.16232917e+00 -1.05591478e+03]
 [ 1.20100599e-04 -2.07510753e-04  1.00000000e+00]]
"""