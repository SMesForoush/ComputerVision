# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
cam = cv2.VideoCapture("../inputs/video.mp4")
output_path = "inputs/frame-{}.jpg"

currentframe = 1
status = True

while status:
    status, frame = cam.read()
    name = output_path.format(currentframe)
    print('Creating...' + name)
    cv2.imwrite(name, frame)
    currentframe += 1
    if currentframe > 900:
        break
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
