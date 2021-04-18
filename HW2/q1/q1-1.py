import math
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from icecream import ic

output_path = '../outputs/{}'
main_indexes = [90, 180, 270, 450, 630, 810]


def img_resize(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output


def load_image(image_address):
    img = cv2.imread(image_address)
    return img


def draw_match(base_image, position1, position2, color, r=5, thickness=5):
    cv2.circle(base_image, position1, r, color, thickness)
    cv2.circle(base_image, position2, r, color, thickness)
    cv2.line(base_image, position1, position2, color, thickness)


def draw_inlier_outlier(draw_img, based_shape, matches, masks, img1_points, img2_points,
                        inlier_color=(0, 0, 255), outlier_color=(255, 0, 0)):
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
    new_img[:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2
    return new_img


def compute_matching_points(img1, img2):
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


def transpose_image(dst_index, start_index):
    print("in transpose_image : ", dst_index, start_index)
    dst_image = load_image(frame_path.format(dst_index))
    start_image = load_image(frame_path.format(start_index))
    match_points = compute_matching_points(dst_image, start_image)
    image_1_points = match_points[0]
    image_2_points = match_points[1]
    homography, mask = cv2.findHomography(image_2_points, image_1_points, cv2.RANSAC, maxIters=7000,
                                          ransacReprojThreshold=100)
    return homography


def continual_homography_upper(start_index, dst_index):
    h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    index = 0
    while main_indexes[index] <= start_index:
        index += 1
    transposing_index = start_index
    print("starting main index : ", index)
    while main_indexes[index] != dst_index and index < len(main_indexes):
        h1 = transpose_image(main_indexes[index], transposing_index)
        h = np.matmul(h, h1)
        transposing_index = main_indexes[index]
        index += 1
    h1 = transpose_image(dst_index, main_indexes[index - 1])
    h = np.matmul(h, h1)
    return h


def continual_homography_lower(start_index, dst_index):
    h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    index = len(main_indexes) - 1
    while main_indexes[index] >= start_index:
        index -= 1
    transposing_index = start_index
    print("starting main index : ", index)
    while main_indexes[index] != dst_index and index < len(main_indexes):
        h1 = transpose_image(main_indexes[index], transposing_index)
        h = np.matmul(h, h1)
        transposing_index = main_indexes[index]
        index -= 1
    h1 = transpose_image(dst_index, main_indexes[index + 1])
    h = np.matmul(h, h1)
    return h


def convert_image(dest_index, start_index):
    if abs(start_index - dest_index) <= 180:
        h = transpose_image(dest_index, start_index)
    else:
        if dest_index > start_index:
            h = continual_homography_upper(start_index, dest_index)
        else:
            h = continual_homography_lower(start_index, dest_index)
    return h


def draw_pts_with_img(img, points, filename):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = np.vstack((points, points[0, :]))
    plt.plot(points[:, 0], points[:, 1])
    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':
    dst_frame = 450
    src_frame = 270
    frame_path = "../inputs/frame-{}.jpg"
    img1 = load_image(frame_path.format(dst_frame))
    img2 = load_image(frame_path.format(src_frame))
    result_size = (img1.shape[1] * 3, img1.shape[0] * 3)
    homography = convert_image(dst_frame, src_frame)
    homography = homography.astype(np.float64)
    points = [(350, 780),
              (860, 780),
              (860, 1000),
              (350, 1000)
              ]
    ic(points)
    start_array = np.array(points, dtype=np.float).reshape((-1, 1, 2))
    result_array = cv2.perspectiveTransform(src=start_array, m=np.linalg.inv(homography))
    image = cv2.polylines(img1.copy(), [start_array.astype(np.int32)],
                          True, (0, 0, 255),
                          5)
    cv2.imwrite(f"result-rect-{dst_frame}.jpg", image)
    image = cv2.polylines(img2.copy(), [result_array.astype(np.int32)],
                          True, (0, 0, 255),
                          5)
    cv2.imwrite(f"result-rect-{src_frame}.jpg", image)

    # cv2.circle(img1, start_point, 5, (255, 0, 0), 5)
    # cv2.imwrite("circle-p.jpg", img1)
    # cv2.circle(img2, (int(result_array[0, 0]), int(result_array[0, 1])), 5, (255, 0, 0), 5)
    # cv2.imwrite("circle2-p.jpg", img2)

    transforming_img = img2.copy()
    t = np.array(
        [[1, 0, result_size[0] / 2 - img1.shape[1] / 2], [0, 1, result_size[1] / 2 - img1.shape[0] / 2], [0, 0, 1]],
        dtype=np.float32)
    h_m = np.matmul(t, homography)
    projected_img = cv2.warpPerspective(transforming_img, h_m, result_size)
    transformed_img = cv2.warpPerspective(img1, t, result_size)
    condition = transformed_img > 0
    projected_img[condition] = transformed_img[condition]
    cv2.imwrite(f"{src_frame}-{dst_frame}-panorama-resized.jpg", projected_img)
