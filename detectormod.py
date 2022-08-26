import cv2
import mediapipe as mp
import time
import numpy as np


class HandDetector:
    def __init__(self, mode=False, detectionCon=0.5, trackCon=0.5):

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=mode,
                                        max_num_hands=1,
                                        min_detection_confidence=detectionCon,
                                        min_tracking_confidence=trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.fingers = {"Thumb": {"CMC": 1, "MCP": 2, "IP": 3, "TIP": 4},
                        "Index": {"MCP": 5, "PIP": 6, "DIP": 7, "TIP": 8},
                        "Middle": {"MCP": 9, "PIP": 10, "DIP": 11, "TIP": 12},
                        "Ring": {"MCP": 13, "PIP": 14, "DIP": 15, "TIP": 16},
                        "Pinky": {"MCP": 17, "PIP": 18, "DIP": 19, "TIP": 20}}

    def find_hand(self, image, draw=True):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_image)

        if self.results.multi_hand_landmarks:
            self.first_hand = self.results.multi_hand_landmarks[0]
            if draw:
                self.mpDraw.draw_landmarks(image, self.first_hand, self.mpHands.HAND_CONNECTIONS)

        return image

    def get_landmark_position(self, landmark_index):

        x = self.first_hand.landmark[landmark_index].x
        y = self.first_hand.landmark[landmark_index].y
        z = self.first_hand.landmark[landmark_index].z

        return [x, y, z]

    def get_distance(self, point1, point2, d3_mode=False):

        x1, y1, z1 = self.get_landmark_position(point1)
        x2, y2, z2 = self.get_landmark_position(point2)

        distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

        if d3_mode:
            distance = (distance ** 2 + (z2 - z1) ** 2) ** 0.5

        return distance

    def get_opened_fingers(self):
        if self.results.multi_hand_landmarks:

            self.opened_fingers = []

            for finger in self.fingers:

                if finger == "Thumb":
                    d1 = self.get_distance(4, 13)
                    d2 = self.get_distance(4, 5)

                else:
                    p1 = self.fingers[finger]["TIP"]
                    p2 = self.fingers[finger]["PIP"]

                    d1 = self.get_distance(p1, 0)
                    d2 = self.get_distance(p2, 0)

                if d1 < d2:
                    self.opened_fingers.append(0)
                else:
                    self.opened_fingers.append(1)

    def find_action(self):

        action_type, action_details = None, None

        if self.results.multi_hand_landmarks:

            self.get_opened_fingers()

            if self.get_distance(8, 12) < 1.5 * self.get_distance(5, 9):
                action_type = "Move"
                action_details = self.get_landmark_position(8)

            elif self.opened_fingers == [1, 1, 0, 0, 0]:
                action_type = "Zoom"
                action_details = self.get_distance(4, 8)

        return action_type, action_details


def main():
    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        img = detector.find_hand(img)

        action_type, _ = detector.find_action()

        cv2.putText(img, f"Action detected: {action_type}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0), 3)

        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime

        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 3)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        # Press 'q' to exit the program.
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
