import string

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model_v2.h5", "Model/labels_v2.txt")
offset = 20
imgSize = 300

folder = "Data/"

counter = 0

#labels = ["A", "B", "C"]
labels = list(string.ascii_uppercase);

frames_predicated = [0]*26;
frames_predicated_counter=0;
predicted_text=""

def zerolistmaker(n):
    return [0] * n

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands = detector.findHands(img, draw=False)
    if frames_predicated_counter>15:
        frames_predicated_counter=0
        index_max = np.argmax(frames_predicated)
        predicted_text+=labels[index_max]
        print(predicted_text)
        frames_predicated = zerolistmaker(26)


    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        aspectRatio = h/w
        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
                # print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hGap + hCal, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
                # print(prediction, index)


        except Exception as e:
            print(e)

        try:
            cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0 ,255), 2)
            frames_predicated[index]+=1
            frames_predicated_counter+=1
            #cv2.imshow("ImageCrop", imgCrop)

            cv2.imshow("ImageWhite", imgWhite)
        except Exception as e:
            print(e)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
