import cv2
import os
import numpy as np

# Folder containing images
folder_path = './'
# Output video filename
video_name = 'output_video.avi'

images = [img for img in os.listdir(folder_path) if img.endswith(".png") or img.endswith(".png")]
images.sort()

frame = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(folder_path, image)))

cv2.destroyAllWindows()
video.release()