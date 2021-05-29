import cv2
import numpy as np
from matplotlib import pyplot as plt
if __name__ == "__main__":
    kernel = np.ones((5, 5), np.uint8)
    for i in range(1, 901):
        frame = cv2.imread(f"../inputs/frame-{i}.jpg")
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
