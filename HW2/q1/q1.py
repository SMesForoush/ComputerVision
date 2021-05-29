BASE_OUTPUTS = "outputs/"
FRAME_CORNER = BASE_OUTPUTS + "frames_corners-new.csv"
FRAME_HOMOGRAPHY = BASE_OUTPUTS + "frames_h-new.csv"

import pandas
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

output_path = 'outputs/{}'
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


def compute_matching_points(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    new_matches = []
    for (m, n) in matches:
        if m.distance < 0.6 * n.distance:
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
    homography, mask = cv2.findHomography(image_2_points, image_1_points, cv2.RANSAC, maxIters=2000)
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


def q11():
    dst_frame = 450
    src_frame = 270
    frame_path = "inputs/frame-{}.jpg"
    img1 = load_image(frame_path.format(dst_frame))
    img2 = load_image(frame_path.format(src_frame))
    result_size = (img1.shape[1] * 3, img1.shape[0] * 3)
    print(img1)
    print(img2)
    homography = convert_image(dst_frame, src_frame)
    homography = homography.astype(np.float64)
    points = [(350, 780),
              (860, 780),
              (860, 1000),
              (350, 1000)
              ]
    print(points)
    start_array = np.array(points, dtype=np.float).reshape((-1, 1, 2))
    result_array = cv2.perspectiveTransform(src=start_array, m=np.linalg.inv(homography))
    image = cv2.polylines(img1.copy(), [start_array.astype(np.int32)],
                          True, (0, 0, 255),
                          5)
    cv2.imwrite(f"‫‪res01-450-rect.jpg‬‬", image)
    image = cv2.polylines(img2.copy(), [result_array.astype(np.int32)],
                          True, (0, 0, 255),
                          5)
    cv2.imwrite(f"‫‪res02-270-rect.jpg‬‬", image)

    transforming_img = img2.copy()
    t = np.array(
        [[1, 0, result_size[0] / 2 - img1.shape[1] / 2], [0, 1, result_size[1] / 2 - img1.shape[0] / 2], [0, 0, 1]],
        dtype=np.float32)
    h_m = np.matmul(t, homography)
    projected_img = cv2.warpPerspective(transforming_img, h_m, result_size)
    transformed_img = cv2.warpPerspective(img1, t, result_size)
    condition = transformed_img > 0
    projected_img[condition] = transformed_img[condition]
    cv2.imwrite(f"‫‪res03-270-450-panorama.jpg‬‬", projected_img)


    # result_size = (img_shape[1] * 3, img_shape[0] * 3)
    # result_image = np.zeros(img_shape, dtype=np.float)
    # t = np.array(
    #     [[1, 0, result_size[0] / 2 - img_shape[1] / 2], [0, 1, result_size[1] / 2 - img_shape[0] / 2], [0, 0, 1]],
    #     dtype=np.float32)
    # mask = np.zeros(result_size, dtype=np.float)
    # for frame in main_indexes:
    #     src_img = load_image(frame_path.format(frame))
    #     homography = convert_image(450, frame)
    #     h_m = np.matmul(t, homography)
    #     projected_img = cv2.warpPerspective(src_img, h_m, result_size)
    #     condition = result_image[:, :, 0] > 0
    #     mask[:, :, :] = 0
    #     mask[condition] = 1
    #     projected_img[condition] = transformed_img[condition]


def compute_key_frames():
    arr = []
    main_keys_dict = {}
    dst_frame = 450
    for src_frame in main_indexes:
        homography = convert_image(dst_frame, src_frame)
        homography = homography.astype(np.float64)
        new_h = homography.reshape((-1)).tolist()
        new_h.insert(0, src_frame)
        arr.append(new_h)
        main_keys_dict[f"{src_frame}-{dst_frame}"] = homography
    df = pandas.DataFrame(arr, columns=["frame", "h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32", "h33"])
    df.to_csv("key_frames.csv")
    return main_keys_dict


def find_homography_with_image(dst_image, start_image):
    match_points = compute_matching_points(dst_image, start_image)
    image_1_points = match_points[0]
    image_2_points = match_points[1]
    homography, mask = cv2.findHomography(image_2_points, image_1_points, cv2.RANSAC, maxIters=7000,
                                          ransacReprojThreshold=100)
    return homography


def find_frame_homography(start_index, start_image, dst_image, key_frame_h):
    dest_index = 450
    if abs(start_index - dest_index) <= 180:
        h = find_homography_with_image(dst_image, start_image)
    else:
        if dest_index > start_index:
            index = 0
            while main_indexes[index] <= start_index:
                index += 1
        else:
            index = len(main_indexes) - 1
            while main_indexes[index] >= start_index:
                index -= 1
        # print(index, main_indexes[index])
        h1 = find_homography_with_image(cv2.imread(frame_path.format(main_indexes[index])), start_image)
        h = np.matmul(h1, key_frame_h.get(f"{main_indexes[index]}-450"))
    return h


def q13():
    key_frames = compute_key_frames()
    dst_image = cv2.imread(frame_path.format(450))
    result_size = (dst_image.shape[1] * 3, dst_image.shape[0] * 3)
    t = np.array(
        [[1, 0, result_size[0] / 2 - dst_image.shape[1] / 2],
         [0, 1, result_size[1] / 2 - dst_image.shape[0] / 2], [0, 0, 1]],
        dtype=np.float32)
    homographies = []
    corner_values = []
    columns = ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32", "h33"]
    corner_columns = ["left x", "left y", "right x", "right y"]
    for i in range(1, 901):
        print("current frame transforming : ", i)
        start_image = cv2.imread(frame_path.format(i))
        h = find_frame_homography(i, start_image, dst_image, key_frames)
        h_m = np.matmul(t, h)
        homographies.append(h_m.reshape((-1)).tolist())
        projected_img = cv2.warpPerspective(start_image, h_m, result_size)
        b, g, r = cv2.split(projected_img)
        y, x = np.where((b > 0) | (g > 0) | (r > 0))
        min_loc = x == x.min()
        max_loc = x == x.max()
        minimum_x = x[min_loc][0]
        minimum_y = y[min_loc][0]
        maximum_x = x[max_loc][0]
        maximum_y = y[max_loc][0]
        corner_values.append([minimum_x, minimum_y, maximum_x, maximum_y])
        cv2.imwrite(f"outputs/frames/frame-{i}.jpg", projected_img)
    df = pandas.DataFrame(homographies, columns=columns)
    df.to_csv(FRAME_HOMOGRAPHY, index=True)
    df = pandas.DataFrame(corner_values, columns=corner_columns)
    df.to_csv(FRAME_CORNER, index=True)
    """
    ffmpeg -f image2 -i frame-%d.jpg result.mp4
    rm frame-*
    """


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
        print(frame_index)
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
        result_corners = result_corners.astype(np.int32)
        cv2.fillPoly(new_border, [result_corners], color=1)
        # plt.imshow(new_border)
        # plt.show()
        if np.count_nonzero(total_image) == 0:
            print("in if")
            total_image = projected_img.copy()
            total_border = new_border.copy()
        else:
            # new_borderprojected_img > 0
            new_border[:, :] = 0
            new_border[projected_img[:, :, 0] > 0] = 1
            total_border[:, :] = 0
            total_border[total_image[:, :, 0] > 0] = 1
            kernel = np.ones((5, 5), np.uint8)
            common_surface = new_border*total_border
            common_surface = cv2.erode(common_surface, kernel, iterations=1)
            blurred = cv2.GaussianBlur(common_surface, (5, 5), 0)
            cv2.line(blurred, (result_corners[0, 0, 0]+5, result_corners[0, 0, 1]+5), (result_corners[1, 0, 0]+5, result_corners[1, 0,  1] - 5), 1, thickness=10)
            condition = blurred > 0
            used_blured = cv2.merge((blurred, blurred, blurred))
            total_image[condition] = (used_blured * total_image)[condition] + ((1-used_blured)*projected_img)[condition]
            total_image[total_image == 0] = projected_img[total_image == 0]
            # total_image[condition] = 255
            # plt.imshow(total_image)
            # plt.show()
            merged_borders = total_border + new_border
            merged_borders[merged_borders > 1] = 1
            # print(merged_borders)
            gradient = cv2.morphologyEx(merged_borders, cv2.MORPH_GRADIENT, kernel)
            total_border = merged_borders
        # plt.imshow(total_border)
        # plt.show()
        new_border[:, :] = 0
    cv2.imwrite("res04.jpg", total_image)


def read_patch(result, frame_corners, frame_indexes, start_x, end_x):
    print(f"in read patch with start x : {start_x} and end x : {end_x}")
    # first image is the one with smallest left x value
    # last image is the one with largest right value
    frame_mask = (frame_corners['left x'] < end_x) & (frame_corners['right x'] > start_x)
    frame_path = "outputs/frames/frame-{}.jpg"

    pixels = {}
    current_frames = frame_corners[frame_mask]
    current_indexes = frame_indexes[frame_mask]
    start_index = current_indexes[0]
    last_index = current_indexes[-1]
    print(start_index, last_index)
    index = start_index
    loop_num = 0
    while index <= last_index:
        # print(frame_path)
        frame_x = cv2.imread(frame_path.format(index+1))
        frame_cropped = frame_x[:, start_x:end_x]
        b, g, r = cv2.split(frame_cropped)
        ys, xs = np.where((b > 3) & (g > 3) & (r > 3))
        if len(xs) == 0:
            if loop_num >= 30:
                print("break loop")
                break
        else:
            loop_num += 1

        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            current = pixels.get(f"{x}-{y}")
            if current:
                frame_arr = (frame_cropped[y, x, :]).tolist()
                current.append(frame_arr)
            else:
                current = [frame_cropped[y, x, :].tolist()]

            pixels[f"{x}-{y}"] = current
        index += 1

    for key, value in pixels.items():
        x, y = key.split("-")
        maximum_color = np.median(np.array(value), axis=0)
        result[int(y), start_x+int(x), :] = maximum_color


def q14():
    image1 = cv2.imread("outputs/frame-1.jpg")
    w, h, c = image1.shape
    result = np.zeros(image1.shape, dtype=np.uint8)
    print(w, h)
    frame_corners = pd.read_csv("outputs/frames_corners-new.csv")
    frame_indexes = frame_corners.index
    patch_size = 100
    left = patch_size+100
    i = 0
    while left < h:
        current_time = time.time()
        read_patch(result, frame_corners, frame_indexes, left-patch_size, left)
        left = left + patch_size
        i += 1
        end_time = time.time()
        diff = (end_time-current_time)/60
        print(f"round i : {i} with diff : {diff}")
    cv2.imwrite("res06-background-panaroma.jpg", result)


def q15():
    background = cv2.imread("res06-background-panaroma.jpg")
    frame = cv2.imread("inputs/frame-1.jpg")
    w, h, c = frame.shape
    frame_hs = pd.read_csv("outputs/frames_h.csv")
    frame_hs.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(frame_hs)
    vals = frame_hs.values
    print(vals[0])
    print(len(vals))
    for i in range(len(vals)):
        flat_h = vals[i]
        print(flat_h)
        current_h = flat_h.reshape((-1, 3))
        print(current_h)
        reverse_h = np.linalg.inv(current_h)
        projected_img = cv2.warpPerspective(background, reverse_h, (h, w))
        cv2.imwrite(f"outputs/bg/background-frame-{i+1}.jpg", projected_img)


def q16():
    kernel = np.ones((5, 5), np.uint8)
    for i in range(1, 901):
        frame = cv2.imread(f"inputs/frame-{i}.jpg")
        background = cv2.imread(f"outputs/bg/background-frame-{i}.jpg")
        new_images = frame.copy()
        frame_f = frame.astype(np.float)
        background_f = background.astype(np.float)
        background_lap = cv2.Laplacian(background, cv2.CV_64F)
        bg = np.zeros(frame.shape[:2])
        bg_mask = (background_lap[:, :, 0] > 50) & (background_lap[:, :, 1] > 50) & (background_lap[:, :, 2] > 50)
        bg[bg_mask] = 1
        bg = cv2.dilate(bg, kernel, iterations=1)

        frame_diff = (frame_f[:, :, 0] - background_f[:, :, 0])**2 + (frame_f[:, :, 1] - background_f[:, :, 1])**2 + (frame_f[:, :, 2] - background_f[:, :, 2])**2
        foreground_mask = frame_diff > 30000

        mask = np.logical_not(bg) & foreground_mask
        r = new_images[:, :, 2]
        g = new_images[:, :, 1]
        b = new_images[:, :, 0]
        r[mask] = 255
        b[mask] = b[mask] - 100
        g[mask] = g[mask] - 100
        cv2.imwrite(f"outputs/fg/foreground-{i}.jpg", new_images)


def q17():
    background = cv2.imread("res06-background-panaroma.jpg")
    frame = cv2.imread("inputs/frame-1.jpg")
    w, h, c = frame.shape
    frame_hs = pd.read_csv(FRAME_HOMOGRAPHY)
    frame_corners = pd.read_csv(FRAME_CORNER)
    frame_hs.drop(['Unnamed: 0'], axis=1, inplace=True)
    vals = frame_hs.values
    img_index = 1
    for i in range(len(vals)):
        flat_h = vals[i]
        current_h = flat_h.reshape((-1, 3))
        reverse_h = np.linalg.inv(current_h)
        transforming_image = np.array(
            [[1, 0, h/4],
             [0, 1, 0], [0, 0, 1]],
            dtype=np.float32)
        final_hom = np.matmul(transforming_image, reverse_h)
        shape = (int(h+h/2), int(w))
        # print(shape)
        projected_img = cv2.warpPerspective(background, final_hom, shape)
        b, g, r = cv2.split(projected_img)
        condition = (b == 0) & (g == 0) & (r == 0)
        blank_img = projected_img[condition]
        print(blank_img.shape)
        if blank_img.shape[0] <= 100:
            cv2.imwrite(f"outputs/bigger/BG-{img_index}.jpg", projected_img)
            img_index += 1

import pandas
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


# def compute_f(homography, w):
    # homography = homography / homography[2, 2]
    # z = math.atan(-1*(homography[0, 1]/homography[1, 1]))
    # cosz = math.cos(z)
    # a = math.sin(z)*(homography[2, 0]/homography[0, 1])
    # temp = math.pow( (2*math.cos(z) - w*a*homography[1,1]) / (2*homography[1,1]) , 2)
    # # siny_f = (-homography[2, 0] * cosz)/homography[1, 1]
    # # cosy = cosz/homography[1, 1] + (w/2)*((homography[2, 0]*cosz)/homography[1, 1])
    # try:
    #     #
    #     # siny = math.sqrt(1 - cosy ** 2)
    #     # y = math.acos(cosy)
    #     f = abs(math.sqrt(1 - temp) / a)
    #     y = a*f
    # except Exception:
    #     return 0
    # return f


def compute_homography_matrix(camera_matrix, x, y, z):
    rotation_matrix = compute_rotation_matrix(x, y, z)
    homography = np.matmul(camera_matrix, np.matmul(rotation_matrix, np.linalg.inv(camera_matrix)))
    return homography


def compute_angels(homography, camera_matrix):
    rotation_matrix = np.matmul(np.linalg.inv(camera_matrix), np.matmul(homography, camera_matrix))
    return compute_euler_angels(rotation_matrix)


def q18():
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


if __name__ == '__main__':
    frame_path = "inputs/frame-{}.jpg"
    # q11()
    # print("finished q1")
    # q12()
    # print("finished q2")
    # q13()
    # print("finished q3")
    # q14()
    print("finished q4")
    q15()
    print("finished q5")
    q16()
    print("finished 6")
    q17()
    print("finished q7")
    q18()


