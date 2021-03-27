import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from skimage.feature import corner_peaks
output_path = '../outputs/{}'


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


def show_image(image, name):
    # convert = cv2.convertScaleAbs(image)
    # plt.imshow(convert)
    # plt.show()
    # cv2.imshow(f"{name}-converted", convert)
    # cv2.waitKey(0)
    cv2.imshow(f"{name}-not converted", image)
    cv2.waitKey(0)
    # cv2.imwrite(f"{name}-converted.jpg", convert)


def write_image(img, name):
    abs_grad = cv2.convertScaleAbs(img)
    cv2.imwrite(output_path.format(name), abs_grad)


def compute_image_gradients(img, i):
    k_size = 15
    s_d = 0
    img_ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    img_iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    img_ixy = img_ix*img_iy
    img_ix2 = img_ix ** 2
    img_iy2 = img_iy ** 2
    grad = np.power(img_iy2+img_ix2, 0.5)
    # abs_grad = cv2.convertScaleAbs(grad)
    # cv2.imwrite(output_path.format(f"res0{i}_grad.jpg"), abs_grad)
    write_image(grad, f"res0{i}_grad.jpg")
    i += 2
    sx2 = cv2.GaussianBlur(img_ix2, (k_size, k_size), sigmaX=s_d)
    sy2 = cv2.GaussianBlur(img_iy2, (k_size, k_size), sigmaX=s_d)
    sxy = cv2.GaussianBlur(img_ixy, (k_size, k_size), sigmaX=s_d)
    sxy2 = np.power(sxy, 2)
    det = sx2*sy2-sxy2
    trace = sx2+sy2
    trace_2 = np.power(trace, 2)
    k = 0.01
    # k = np.max(det) / np.max(trace_2)
    print("det -> max : ", np.max(det), " min : ", np.min(det))
    print("trace -> max : ", np.max(trace_2), " min : ", np.min(trace_2))
    print("k : ", k)
    R = det - k * trace_2
    write_image(R, f"res0{i}_score.jpg")
    i += 2
    print("R -> max : ", np.max(R), " min : ", np.min(R))
    threshold = np.max(R)/500000
    # threshold = 12500
    print(threshold)
    R[R < threshold] = 0
    write_image(R, f"res0{i}_thresh.jpg")
    i += 2
    r_scaled = cv2.convertScaleAbs(R)
    r_scaled = cv2.cvtColor(r_scaled, cv2.COLOR_BGR2GRAY)
    # ones = np.full((R.shape[0], R.shape[1]), 255, dtype=np.uint8)
    # condition = (R[:, :, 0] < threshold) & (R[:, :, 1] < threshold) & (R[:, :, 2] < threshold)
    # ones[condition] = 0
    # x, y = np.where(ones > 0)
    # w = np.zeros(x.shape)
    # w = x + 20
    # h = np.zeros(y.shape)
    # h = y + 20
    # boxes = np.vstack((x, y, w, h)).T
    coords = corner_peaks(r_scaled, min_distance=20, threshold_rel=0.5)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.plot(coords[:, 1], coords[:, 0], color='red', marker='o',
            linestyle='None', markersize=1)
    plt.axis("off")
    plt.savefig(output_path.format(f"res0{i}_harris.jpg"))

    # print(boxes)

    """features vector"""
    n = 10
    start_x = (coords[:, 0] - n/2).astype(np.int)
    end_x = (coords[:, 0] + n/2).astype(np.int)
    start_y = (coords[:, 1] - n/2).astype(np.int)
    end_y = (coords[:, 1] + n/2).astype(np.int)
    features = np.zeros((start_x.shape[0], n**2, 3))
    for i in range(start_x.shape[0]):
        feature_window = img[start_x[i]:end_x[i], start_y[i]:end_y[i], :].copy()
        feature_array = feature_window.reshape((-1, 3))
        features[i, :, :] = feature_array
    return features, coords


def compute_cores(features1, cores1, features2):
    for q in range(cores1.shape[0]):
        feature = features1[q, :, :]
        dist = (features2[:] - feature)**2
        distances = np.sum(dist, axis=(2, 1))
        d1, d2 = np.partition(distances, 1)[:2]
        # print(d1, d2)
        diff = d2/d1
        # print(diff)
        if diff >= 2.5:
            # print("in if")
            cores1[q] = np.argwhere(distances == d1)
        else:
            cores1[q] = -1
    return cores1


if __name__ == '__main__':
    img1 = load_image("../inputs/im01.jpg")
    print(img1.shape)
    img2 = load_image("../inputs/im02.jpg")
    features_1, cords_1 = compute_image_gradients(img1, 1)
    features_2, cords_2 = compute_image_gradients(img2, 2)
    cores_1 = np.zeros(cords_1.shape[0], dtype=np.int)
    cores_2 = np.zeros(cords_2.shape[0], dtype=np.int)
    cores_1 = compute_cores(features_1, cores_1, features_2)
    cores_2 = compute_cores(features_2, cores_2, features_1)

    # """removing duplicate"""
    # sorted_cros = np.sort(cores_1)
    # for q in range(sorted_cros.shape[0]-1):
    #     if sorted_cros[q] == sorted_cros[q+1]:
    #         cores_1[cores_1 == sorted_cros[q]] = -1
    # sorted_cros = np.sort(cores_2)
    # for q in range(sorted_cros.shape[0]-1):
    #     if sorted_cros[q] == sorted_cros[q+1]:
    #         cores_2[cores_2 == sorted_cros[q]] = -1
    final = []

    res = np.argwhere(cores_1 > 0)
    for point in res:
        q1_add = cores_1[point]
        # print(q1_add, cores_2[q1_add[0]], point)
        if cores_2[q1_add[0]] == point[0]:
            final.append([point[0], q1_add[0]])
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1)
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2)
    # plt.axis("off")
    for p1, q1 in final:
        x = cords_1[p1, 1]
        y = cords_1[p1, 0]
        rgb = np.random.rand(3, )
        print(rgb)
        ax1.plot(x, y, 'x', c=rgb, markersize=5)
        y2, x2 = cords_2[q1]
        ax2.plot(x2, y2, 'x', c=rgb, markersize=5)
        con = ConnectionPatch(xyA=(x2, y2), xyB=(x, y), coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color=rgb)
        ax2.add_artist(con)
    plt.show()
