import math
import random

import cv2
import numpy as np
import pandas as pd
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
        if m.distance < 0.5 * n.distance:
            new_matches.append(m)

    image_1_points = np.zeros((len(new_matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(new_matches), 1, 2), dtype=np.float32)

    for i in range(len(new_matches)):
        image_1_points[i] = kp1[new_matches[i].queryIdx].pt
        image_2_points[i] = kp2[new_matches[i].trainIdx].pt
    print("matching points len : ", len(image_1_points))
    return image_1_points, image_2_points


def transpose_image(dst_index, start_index):
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
    while main_indexes[index] != dst_index and index < len(main_indexes):
        h1 = transpose_image(main_indexes[index], transposing_index)
        h = np.matmul(h, h1)
        transposing_index = main_indexes[index]
        index -= 1
    h1 = transpose_image(dst_index, main_indexes[index + 1])
    h = np.matmul(h, h1)
    return h


def convert_image(dest_index, start_index):
    if start_index == dest_index:
        return np.eye(3, 3)
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


def q11():
    dst_frame = 450
    src_frame = 270
    frame_path = "inputs/frame-{}.jpg"
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


def compute_min_path(patch_1, patch_2, starting_point, ending_point):
    patch_1 = np.array(patch_1, dtype=np.float32)
    patch_2 = np.array(patch_2, dtype=np.float32)
    diff = (patch_1-patch_2)**2
    feature = diff[:, :, 0]+diff[:, :, 1]+diff[:, :, 2]
    r1, g1, b1 = cv2.split(patch_1)
    r2, g2, b2 = cv2.split(patch_2)
    condition = ((r1 == 0) & (b1 == 0) & (g1 == 0)) | ((r2 == 0) & (b2 == 0) & (g2 == 0))
    feature[condition] = 200000
    plt.subplot(1, 2, 1)
    plt.title("feature without starting point")
    plt.imshow(feature)
    feature[:starting_point[1], :] = 200000
    feature[starting_point[1], starting_point[0]] = 0
    plt.subplot(1, 2, 2)
    plt.title("feature with starting point")
    plt.imshow(feature)
    plt.show()
    # cv2.imshow(feature)
    # print(feature.dtype)
    # print(f"min feature {np.min(feature[0, :])}")
    minimum_path = np.full(feature.shape, np.inf)
    shape = (int(feature.shape[0]), int(feature.shape[1]), 2)
    last_paths = np.zeros(shape)
    min_j = min(starting_point[0], ending_point[0])
    for i in range(minimum_path.shape[1]):
        minimum_path[starting_point[1], i] = feature[starting_point[1], i]
    for i in range(starting_point[1], feature.shape[0]):
        for j in range(0, feature.shape[1]):
            # cv2.circle(patch_2, (j, i), 5, (255, 0, 0), 5)
            min_path = np.inf
            for n in range(-2, 5):
                if 0 <= j+n < feature.shape[1]:
                    path = feature[i, j] + minimum_path[i-1, j+n]
                    if min_path > path:
                        min_path = path
                        minimum_path[i, j] = path
                        last_paths[i, j, :] = [i - 1, n+j]
        # if i % 300 == 0:
        #     plt.subplot(1, 2, 1)
        #     plt.title("current point")
        #     plt.imshow(patch_2)

    last_point = ending_point
    # final_path = np.inf
    # for i in range(feature.shape[1]):
    #     if minimum_path[feature.shape[0]-1, i] < final_path:
    #         last_point[1] = i
    #         final_path = minimum_path[feature.shape[0]-1, i]
    paths = [last_point, ]
    for i in range(2, feature.shape[0]):
        last_point = last_paths[int(last_point[0]), int(last_point[1])]
        cv2.circle(patch_2, (int(last_point[0]), int(last_point[1])), 10, (255, 0, 0), 10)
        paths.append(last_point)
        print(last_point)
    print(paths)
        # cv2.circle(patch_1, last_point, 2, (255, 255, 255), 2)
    plt.subplot(1, 2, 1)
    plt.title("path")
    plt.imshow(patch_2)
    plt.subplot(1, 2, 2)
    plt.title("min path")
    plt.imshow(minimum_path)
        # plt.show()
    plt.show()

    # print("minimum paths : ")
    # print(minimum_path)
    return paths


def merge_two_frames(total_image, total_border, projected_img, new_border):
    if np.count_nonzero(total_image) == 0:
        print("in if")
        total_image = projected_img.copy()
        total_border = new_border.copy()
    else:
        kernel = np.ones((3, 3), np.uint8)
        nb_grad = cv2.morphologyEx(new_border, cv2.MORPH_GRADIENT, kernel)
        tot_grad = cv2.morphologyEx(total_border, cv2.MORPH_GRADIENT, kernel)
        # plt.subplot(2, 2, 1)
        # plt.imshow(nb_grad)
        # plt.subplot(2, 2, 2)
        # plt.imshow(tot_grad)
        common_points = nb_grad*tot_grad
        xs, ys = np.where(common_points > 0)
        starting_point = (ys[0], xs[0])
        ending_point = (ys[-1], xs[-1])
        print(starting_point, ending_point)
        # cv2.circle(projected_img, starting_point, 5, (255, 0, 0), 2)
        # cv2.circle(projected_img, ending_point, 5, (255, 0, 0), 2)
        # cv2.circle(total_image, ending_point, 5, (255, 0, 0), 2)
        # cv2.circle(total_image, ending_point, 5, (255, 0, 0), 2)
        # plt.subplot(2, 1, 1)
        # plt.imshow(projected_img)
        # plt.subplot(2, 1, 2)
        # plt.imshow(total_image)
        # plt.show()
        path = compute_min_path(total_image, projected_img, starting_point, ending_point)
        # for i, j in path:
        #     i = int(i)
        #     j = int(j)
        #     total_border[i, j:] = 1
        # plt.title("total border")
        # plt.imshow(total_border)
        # plt.show()
        merged_borders = total_border+new_border
        merged_borders[merged_borders > 1] = 1
        # print(merged_borders)
        gradient = cv2.morphologyEx(merged_borders, cv2.MORPH_GRADIENT, kernel)
        # plt.subplot(2, 2, 3)
        # plt.imshow(common_points)
        # plt.subplot(2, 2, 4)
        # plt.imshow(gradient)
        # plt.show()
        total_border = merged_borders
    return total_image, total_border


def q12():
    img_shape = (1080*3, 1920*3, 3)
    result_shape = (1920*3, 1080*3)
    frame_shape = [1920, 1080]
    img_corners = [[0, 0], [0, 1080], [1920, 1080], [1920, 0]]
    total_border = np.zeros(img_shape[:2], dtype=np.float)
    new_border = np.zeros(img_shape[:2], dtype=np.float)
    total_image = np.zeros(img_shape, dtype=np.float)
    # for index in main_indexes:
    #     current_image = cv2.imread(frame_path.format(index))
    main_frames = pd.read_csv("key_frames.csv")
    for index, row in main_frames.iterrows():
        frame_index = row.values[1]
        frame_img = cv2.imread(frame_path.format(int(frame_index)))
        frame_h = row.values[2:]
        homography = frame_h.reshape((-1, 3))
        t = np.array(
            [[1, 0, result_shape[0] / 2 - frame_img.shape[1] / 2], [0, 1, result_shape[1] / 2 - frame_img.shape[0] / 2], [0, 0, 1]],
            dtype=np.float32)
        h_m = np.matmul(t, homography)
        start_array = np.array(img_corners, dtype=np.float).reshape((-1, 1, 2))
        result_corners = cv2.perspectiveTransform(src=start_array, m=h_m)
        projected_img = cv2.warpPerspective(frame_img, h_m, result_shape)

        # cv2.polylines(projected_img, [result_corners.astype(np.int32)], True, (0, 0, 255), 5)
        cv2.fillPoly(new_border, [result_corners.astype(np.int32)], color=1)
        # plt.imshow(new_border)
        # plt.show()
        total_image, total_border = merge_two_frames(total_image, total_border, projected_img, new_border)
        plt.imshow(total_image)
        plt.show()
        # plt.imshow(total_border)
        # plt.show()
        new_border[:, :] = 0
    cv2.imwrite("final_result.jpg", total_image)


# def q12():
#
#     total_image_border =


if __name__ == '__main__':
    frame_path = "inputs/frame-{}.jpg"
    # q11()
    q12()
    # img1 = load_image(frame_path.format(450))
    # print(img1.shape)


