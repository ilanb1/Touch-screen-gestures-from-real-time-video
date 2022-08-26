import cv2
import time
from detectormod import HandDetector
from utils import Image


test_image_path = "test_image.jpg"
test_image = cv2.imread(test_image_path)

shape = (540, 960)
image = Image(frame_shape = shape)
image.update_image(test_image)



detector = HandDetector()
cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

while True:

    success, img = cap.read()
    img = detector.find_hand(img)
    img = cv2.flip(img, 1)

    action_type, action_details = detector.find_action()
    image.perform_action(action_type, action_details)

    result = image.get_frame()

    # Calculating and displaying the FPS:
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)


    cv2.imshow("Image", img)
    cv2.imshow("result", result)

    key = cv2.waitKey(1)
    # Press 'q' to exit the program.
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()