import os
import time

import cv2

import handtrack

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#folder_path = "FingerImages"
#my_list = os.listdir(folder_path)

#overlay_list = []
#for im_path in my_list:
 #   image = cv2.imread(f'{folder_path}/{im_path}')
 #   overlay_list.append(image)

p_time = 0
detector = handtrack.hand_detector()

tip_ids = [4, 8, 12, 16, 20]


while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    #cv2.imshow('image', img)

    lm_list = detector.find_positions(img, draw=False)

    if(len(lm_list) != 0):
        fingers = []

        if(lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]):
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

        # h, w, c = overlay_list[total_fingers - 1].shape
        # img[0:h, 0:w] = overlay_list[total_fingers - 1]

        # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)

        cv2.putText(img, str(total_fingers), (45, 375),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('image', img)
    if(cv2.waitKey(10) == ord('q')):
        break
