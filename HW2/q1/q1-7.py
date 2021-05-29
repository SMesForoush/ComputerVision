import cv2
import numpy as np
import pandas as pd
from q1 import *

if __name__ == '__main__':
    background = cv2.imread("background_subtracted.jpg")
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
            cv2.imwrite(f"outputs/biggerNew/BG-{img_index}.jpg", projected_img)
            img_index += 1
