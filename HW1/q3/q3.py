import cv2
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


def draw_matches(new_img, based_shape, kp1, kp2, matches, color=None):
    r = 5
    thickness = 5
    img_with_line = new_img.copy()
    for i, (m, n) in enumerate(matches):
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([based_shape[1], 0]))
        color = np.random.randint(0,256)

        cv2.circle(new_img, end1, r, color, thickness)
        cv2.circle(new_img, end2, r, color, thickness)
        cv2.circle(img_with_line, end1, r, color, thickness)
        cv2.circle(img_with_line, end2, r, color, thickness)
        cv2.line(img_with_line, end1, end2, color, thickness)
    return new_img, img_with_line


if __name__ == '__main__':
    img1 = load_image("../inputs/im03.jpg")
    img2 = load_image("../inputs/im04.jpg")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    res = cv2.drawMatches(img1, kp1, img2, kp2, [], img2, flags=0, singlePointColor=(0, 255, 0))
    cv2.imwrite(output_path.format("res13_corners.jpg"), res)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    print("max : ", np.max(des1), " min: ", np.min(des1))
    print("max : ", np.max(des2), " min: ", np.min(des2))
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=200)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    print(matches)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    filtered_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.65 * n.distance:
            matchesMask[i] = [1, 0]
            filtered_matches.append([m, n])

    draw_params = dict(matchColor=(0, 0, 0),
                       singlePointColor=(0, 255, 0),
                       matchesMask=matchesMask,
                       flags=2)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    img4, with_line = draw_matches(res, img1.shape, kp1, kp2, filtered_matches, color=(0, 0, 255))
    cv2.imwrite(output_path.format("res14_correspondences.jpg"), img4)
    cv2.imwrite(output_path.format("res15_match.jpg"), with_line)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img3)
    ax2 = fig.add_subplot(122)
    ax2.imshow(img4)
    plt.show()

    # print(kp2)
