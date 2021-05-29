import cv2
import glob
input_path = "/home/sahel/internship/final_result/cropped/*.jpg"
output_path = "/home/sahel/internship/final_result/without_center/{}"
dsize = (256, 256)
images = glob.glob(input_path)
print(images)
for image_path in images:
    name = image_path.split("/")[-1]
    image = cv2.imread(image_path)
    print(image.shape)
    output = cv2.resize(image, dsize)
    cv2.imwrite(output_path.format(name), output)
