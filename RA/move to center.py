import cv2
import numpy as np
import glob


def crop_image(img):
    w, h, c = img.shape
    center = w/2
    b, g, r = cv2.split(img)
    y, x = np.where((b > 20) & (g > 20) & (r > 20))
    minimum_x = x[x == x.min()][0]
    minimum_y = y[y == y.min()][0]
    maximum_x = x[x == x.max()][0]
    maximum_y = y[y == y.max()][0]
    min_value = min(minimum_y, minimum_x)
    max_value = max(maximum_y, maximum_x)
    max_diff = max(abs(center-max_value), abs(center-min_value))
    start = int(center - max_diff)
    end = int(center + max_diff)
    print(min_value, max_value)
    cropped_img = img[start:end, start:end]
    return cropped_img


result_path = "/home/sahel/internship/final_result/cropped/{}"
crop_image(cv2.imread("BAR_20_a1_f5_c25_180L.jpg"))
new_path = "/home/sahel/internship/final_result/without_bars/*.jpg"
all_images = glob.glob(new_path)
for image in all_images:
    cropped = crop_image(cv2.imread(image))
    name = image.split("/")[-1]
    cv2.imwrite(result_path.format(name), cropped)


# t = "/home/sahel/internship/final_result/cropped/BAR_10_a1_f1_c25_180L.jpg"
# cropped = cv2.imread(t)
# print(cropped.shape)