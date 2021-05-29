import cv2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    background = cv2.imread("background_subtracted.jpg")
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
        cv2.imwrite(f"outputs/background-frame-{i+1}.jpg", projected_img)
