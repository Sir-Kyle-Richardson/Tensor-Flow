import math
from tqdm import tqdm
import cv2  # for capturing videos
import os

currentPath = os.path.dirname(os.path.abspath(__file__))

# Splits video by frames into src/tmp folder
def splitVideo(video):
    count = 0
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    frameRate = math.floor(cap.get(5))

    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()

        if ret != True:
            break

        if frameId % frameRate == 0:
            # storing the frames in a new folder named train_1
            filename = currentPath + "/../tmp/frames/video_frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)

    return True
