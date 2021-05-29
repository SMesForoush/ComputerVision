import pandas

from q11 import convert_image
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import math
frame_path = "inputs/frame-{}.jpg"


def compute_euler_angels(matrix):
    r = R.from_matrix(matrix)
    result = r.as_euler("xyz")
    return result


def compute_rotation_matrix(x, y, z):
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]], dtype=np.float32)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]], dtype=np.float32)
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]], dtype=np.float32)
    R = np.matmul(Rz, np.matmul(Ry, Rx))
    return R


def compute_f(homography, w):
    homography = homography / homography[2, 2]
    z = math.atan(-1*(homography[0, 1]/homography[1, 1]))
    cosz = math.cos(z)
    siny_f = (-homography[2, 0] * cosz)/homography[1, 1]
    cosy = cosz/homography[1, 1] + (w/2)*((homography[2, 0]*cosz)/homography[1, 1])
    try:
        siny = math.sqrt(1 - cosy ** 2)
        y = math.acos(cosy)
        f = siny/siny_f
    except Exception:
        return 0
    return f


def compute_homography_matrix(camera_matrix, x, y, z):
    rotation_matrix = compute_rotation_matrix(x, y, z)
    homography = np.matmul(camera_matrix, np.matmul(rotation_matrix, np.linalg.inv(camera_matrix)))
    return homography


def compute_angels(homography, camera_matrix):
    rotation_matrix = np.matmul(np.linalg.inv(camera_matrix), np.matmul(homography, camera_matrix))
    return compute_euler_angels(rotation_matrix)


def create_shakeless_image():
    img = cv2.imread("inputs/frame-1.jpg")
    h,w, c = img.shape
    # one_by_one_h = pd.read_csv("outputs/homographiesonebyone.csv")
    # one_by_one_h = pd.read_csv("outputs/frames_h-new.csv")
    homographies = np.genfromtxt("homographies.csv")
    print(homographies.shape)
    result_shape = (1920*3, 1080*3)
    frame_shape = (1920, 1080)
    # t = np.array(
    #     [[1, 0, result_shape[0] / 2 - frame_shape[0] / 2], [0, 1, result_shape[1] / 2 - frame_shape[1] / 2],
    #      [0, 0, 1]],
    #     dtype=np.float32)
    # trans_inv = np.linalg.inv(t)
    f_s = []
    i = 0
    ys = []
    zs = []
    xs = []
    for index in range(homographies.shape[0]):
        row = homographies[index]
        with_trans = row.reshape((3, 3))
        # print(f"homography i: {index} \n{with_trans}")
        # homography = np.matmul(trans_inv, with_trans)
        f = compute_f(with_trans, w)
        # if i > 1:
        #     y += ys[-1]
        #     z += zs[-1]
        # ys.append(y)
        # zs.append(z)
        if f > 0:
            f_s.append(f)
            i += 1
    f_mean = np.average(f_s)
    plt.title("x rotation angle")
    plt.scatter(range(len(f_s)), f_s, s=1)
    # plt.scatter(range(len(averaged_z)), averaged_z, s=1)
    plt.show()
    # f_mean = 1630
    print(f_mean)
    camera_matrix = np.array([[f_mean, 0, w / 2], [0, f_mean, h / 2], [0, 0, 1]], dtype=np.float32)
    for index in range(0, homographies.shape[0]):
        row = homographies[index]
        homography = row.reshape((3, 3))
        x, y, z = compute_angels(homography, camera_matrix)
        ys.append(y)
        zs.append(z)
        xs.append(x)

    averaged_y = ys.copy()
    averaged_z = zs.copy()
    averaged_x = xs.copy()
    averaged_y = np.array(averaged_y)
    averaged_z = np.array(averaged_z)
    averaged_x = np.array(averaged_x)
    start_window = 15
    end_window = 15
    for i in range(start_window, len(ys)-end_window):
        current_window_y = averaged_y[i-start_window:i+end_window]
        current_window_z = averaged_z[i-start_window:i+end_window]
        current_window_x = averaged_x[i-start_window:i+end_window]
        averaged_y[i] = np.average(current_window_y)
        averaged_z[i] = np.average(current_window_z)
        averaged_x[i] = np.average(current_window_x)
    plt.subplot(3, 1, 1)
    plt.title("y rotation angle")
    plt.scatter(range(len(ys)), ys, s=1)
    plt.scatter(range(len(averaged_y)), averaged_y, s=1)
    plt.subplot(3, 1, 2)
    plt.title("z rotation angle")
    plt.scatter(range(len(zs)), zs, s=1)
    plt.scatter(range(len(averaged_z)), averaged_z, s=1)
    plt.subplot(3, 1, 3)
    plt.title("x rotation angle")
    plt.scatter(range(len(xs)), xs, s=1)
    plt.scatter(range(len(averaged_x)), averaged_x, s=1)
    plt.show()
    print(len(averaged_y))
    for index in range(0, homographies.shape[0]):
        row = homographies[index]
        with_shake_h = row.reshape((3, 3))
        without_shake_h = compute_homography_matrix(camera_matrix, averaged_x[index],
                                                    averaged_y[index], averaged_z[index])
        remove_shake_h = np.matmul(np.linalg.inv(without_shake_h), with_shake_h)
        image = cv2.imread(f"inputs/frame-{index+1}.jpg")
        shakeles_image = cv2.warpPerspective(image, remove_shake_h, frame_shape)
        cv2.imwrite(f"outputs/newunshaked/frame-{index+1}.jpg", shakeles_image)


def create_one_by_one_homography():
    homographies = []
    for i in range(1, 900):
        h = convert_image(i+1, i)
        homographies.append(h.reshape((-1)).tolist())
    df = pandas.DataFrame(homographies, columns=["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32", "h33"])
    df.to_csv("homographiesonebyone.csv")


create_shakeless_image()