import cv2
import numpy as np
INPUT_PATH = "../q1/inputs/im{:02d}.jpg"


def compute_image_points(start_image, end_image):
    chessboard_criteria = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    real_points = []
    image_points = []
    chessboard_dim = (6, 9)
    chessboard_points = chessboard_dim[0] * chessboard_dim[1]
    corner_sub_pix_cri = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objectp3d = np.zeros((1, chessboard_points, 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:chessboard_dim[0], 0:chessboard_dim[1]].T.reshape(-1, 2)
    for i in range(start_image, end_image+1):
        image = cv2.imread(INPUT_PATH.format(i), cv2.IMREAD_GRAYSCALE)
        ret, corners = cv2.findChessboardCorners(image, chessboard_dim, chessboard_criteria)
        if ret:
            real_points.append(objectp3d)
            corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), corner_sub_pix_cri)
            image_points.append(corners2)
    return image_points, real_points


def compute_camera_calibration(start, end, flag):
    twodpoints, threedpoints = compute_image_points(start, end)
    img = cv2.imread(INPUT_PATH.format(1), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    px = h / 2
    py = w / 2
    camera_matrix = np.array([[1, 0, py], [0, 1, px], [0, 0, 1]])

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints,
                                                                  (w, h), camera_matrix, None, flags=flag)
    return matrix


if __name__=="__main__":
    flag = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT
    m1 = compute_camera_calibration(1, 10, None)
    m2 = compute_camera_calibration(6, 15, None)
    m3 = compute_camera_calibration(11, 20, None)
    m4 = compute_camera_calibration(1, 20, None)
    print(m1)
    print(m2)
    print(m3)
    print(m4)
    m5 = compute_camera_calibration(1, 10, flag)
    m6 = compute_camera_calibration(6, 15, flag)
    m7 = compute_camera_calibration(11, 20, flag)
    m8 = compute_camera_calibration(1, 20, flag)
    print(m5)
    print(m6)
    print(m7)
    print(m8)
