import string

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
from os import listdir
import time

#Instructions
#Before ruuning this program make sure you have 'SLR_processed_images' named directory
#Before ruuning this program make sure you have 'asl_alphabet_train' named directory
#This program currentley proccesses 400 images of each alphabet
no_of_images_to_process=400;

offset = 20
imgSize = 300

counter = 0
detector = HandDetector(maxHands=1)

#list of alphabets from A to Z in uppercase
alphabets = list(string.ascii_uppercase);

for alp in alphabets:
    folder_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'asl_alphabet_train'+'/'+alp,''))
    put_image_in = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'SLR_processed_images'+'/'+alp,''))
    counter=0
    if not os.path.exists(put_image_in):
        # Create a new directory because it does not exist
        os.makedirs(put_image_in)
    for images in os.listdir(folder_dir):
        # check if the image ends with jpg
        if (images.endswith(".jpg")):
            img = cv2.imread(folder_dir+'/'+images)
            hands, img = detector.findHands(img)
            try:
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                    imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap: wCal + wGap] = imgResize

                    elif aspectRatio <= 1:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap: hGap + hCal, :] = imgResize

                    else:
                        break


                cv2.imwrite(f'{put_image_in}/Image_{time.time()}.jpg', imgWhite)
                counter = counter + 1
                if counter>no_of_images_to_process:
                    break
            except Exception as e: print("An error occured skipping " + images)