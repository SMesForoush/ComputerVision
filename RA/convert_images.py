import glob

import cv2

resized_path = "/home/sahel/dataset_buddha/buddha/images/{}.jpg"
toy_images = glob.glob("/home/sahel/dataset_buddha/buddha/images/*.png")

i = 1
for image_path in toy_images:
    img = cv2.imread(image_path)
    cv2.imwrite(resized_path.format(i), img)
    i += 1


