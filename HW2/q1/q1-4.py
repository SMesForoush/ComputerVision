import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
frame_path = "outputs/frame-{}.jpg"


def read_patch(start_x, end_x):
    print(f"in read patch with start x : {start_x} and end x : {end_x}")
    # first image is the one with smallest left x value
    # last image is the one with largest right value
    frame_mask = (frame_corners['left x'] < end_x) & (frame_corners['right x'] > start_x)
    pixels = {}
    current_frames = frame_corners[frame_mask]
    for index, row in current_frames.iterrows():
        frame_x = cv2.imread(frame_path.format(index+1))
        frame_cropped = frame_x[:, start_x:end_x]
        b, g, r = cv2.split(frame_cropped)
        ys, xs = np.where((b > 0) & (g > 0) & (r > 0))
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
    for key, value in pixels.items():
        x, y = key.split("-")
        maximum_color = np.median(np.array(value), axis=0)
        result[int(y), start_x+int(x), :] = maximum_color


if __name__ == "__main__":
    image1 = cv2.imread("outputs/frame-1.jpg")
    w, h, c = image1.shape
    result = np.zeros(image1.shape, dtype=np.uint8)
    print(w, h)
    frame_corners = pd.read_csv("outputs/frames_corners.csv")
    patch_size = 100
    left = patch_size+100
    i = 0
    while left < h:
        current_time = time.time()
        read_patch(left-patch_size, left)
        left = left + patch_size
        i += 1
        end_time = time.time()
        diff = (end_time-current_time)/60
        print(f"round i : {i} with diff : {diff}")
    cv2.imwrite("background_subtracted.jpg", result)
