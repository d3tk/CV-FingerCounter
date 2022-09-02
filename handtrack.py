import time

import cv2
import mediapipe as mp


class hand_detector():
    def __init__(self, mode=False, max_hands=2, detection_con=1, track_con=1):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_hands
        )

        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_positions(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y*h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lm_list


def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    detector = hand_detector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_positions(img)
        # if len(lm_list) != 0:
        #     print(lm_list[4])

        c_time = time.time()

        fps = 1/(c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("image", img)
        if(cv2.waitKey(10) == ord('q')):
            break


if __name__ == "__main__":
    main()
