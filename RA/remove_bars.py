import cv2
import glob
import re
import numpy as np


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
    # print(min_value, max_value)
    cropped_img = img[start:end, start:end]
    return cropped_img


input_path = "/home/sahel/internship/new_images/*.jpg"
output_path = "/home/sahel/internship/final_result/without_bars/{}"
dsize = (256, 256)
images = glob.glob(input_path)
print(images)
for image_path in images:
    image = cv2.imread(image_path)
    w, h, c = image.shape
    center = (int(w/2), int(h/2))
    name = image_path.split("/")[-1]
    # print(name)
    try:
        pas = name.split('_')[1]
        pa_list = pas.split('-')
        if len(pa_list) > 1:
            # having bar
            print(f"has bar : {name}")
            bar_size = int(pa_list[0])
            center_size = re.findall(r"c\d+", name)[0][1:]
            center_size = int(center_size)
            # print(center_size, bar_size)
            # print(bar_size)
            if name.startswith("BV"):
                image = cv2.ellipse(image, center, (bar_size+center_size, center_size), 0, 0, 360,
                                    color=(0, 0, 0), thickness=-1)
            elif name.startswith("V"):
                print("name starts with V : ", name)
                image = cv2.ellipse(image, center, (center_size, center_size), 0, 0, 360,
                                    color=(0, 0, 0), thickness=-1)
            else:
                image = cv2.ellipse(image, center, (center_size, center_size), 0, 0, 360,
                                    color=(0, 0, 0), thickness=-1)
            result = crop_image(image)
            # cv2.imshow("removed bar", image)
            # cv2.waitKey(0)
        else:
            result = crop_image(image)

        # print(image.shape)
        output = cv2.resize(result, dsize)
        # print(output.shape)
        cv2.imwrite(output_path.format(name), output)
    except Exception:
        print("exception")

# image_path = "/home/sahel/internship/cropped_images/BVAR_25-10_a2_f5_c25_180L.jpg"
# image_path2 = "/home/sahel/internship/cropped_images/BVAR_50-10_a2_f3_c25_180L.jpg"
# img = cv2.imread(image_path)
# print(img.shape)
# img2 = cv2.imread(image_path2)
# print(img2.shape)
# # name = image_path.split("/")[-1]
# # pas = name.split('_')[1]
# # pa_list = pas.split('-')
# # bar_size = int(pa_list[0])
# # print(bar_size)
# cv2.imshow("b",img )
# cv2.imshow("2b",img2 )
# cv2.waitKey(0)
