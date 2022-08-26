import cv2
import numpy as np


class Image:

    def __init__(self, frame_shape=(480, 640)):

        self.frame_shape = frame_shape[:2]
        self.aspect_ratio = self.frame_shape[1] / self.frame_shape[0]

        # Minimum size of zoom in:
        self.MIN_HEIGHT = 50

        # logs of the real time actions:
        self.move_log = [0, 0]  # previous location.
        self.zoom_log = 0  # previous size

    def update_image(self, image):
        self.original_image = image

        # info of the current ROI (top left corner of the rectangle, and its dimensions) :
        self.top_left_corner = [0, 0]
        self.current_roi_height = self.original_image.shape[0]
        self.current_roi_width = np.clip(a=int(self.current_roi_height * self.aspect_ratio),
                                         a_min=None,
                                         a_max=self.original_image.shape[1])

        # The ROI itself:
        self.update_frame()

    def update_frame(self):

        roi = self.original_image[self.top_left_corner[0]: self.top_left_corner[0] + self.current_roi_height,
              self.top_left_corner[1]: self.top_left_corner[1] + self.current_roi_width]

        self.frame = cv2.resize(roi, self.frame_shape[::-1])

    def get_frame(self):
        return self.frame

    def move(self, x, y):

        # Calculating the new position:
        self.top_left_corner[0] += y
        self.top_left_corner[1] += x

        # Clipping these values to avoid going out of the original image:
        self.top_left_corner[0] = np.clip(a=self.top_left_corner[0],
                                          a_min=0,
                                          a_max=self.original_image.shape[0] - self.current_roi_height)

        self.top_left_corner[1] = np.clip(a=self.top_left_corner[1],
                                          a_min=0,
                                          a_max=self.original_image.shape[1] - self.current_roi_width)

        # Updating the frame:
        self.update_frame()

    def zoom(self, factor):

        # Calculating the new size:
        new_roi_height = int(self.current_roi_height * factor)

        # Clipping this value to avoid reaching the minimum allowed size, or exceed the original image shape:
        new_roi_height = np.clip(a=new_roi_height,
                                 a_min=self.MIN_HEIGHT,
                                 a_max=self.original_image.shape[0])

        new_roi_width = int(new_roi_height * self.aspect_ratio)

        # Calculating the new position:
        self.top_left_corner[0] += (self.current_roi_height - new_roi_height) // 2
        self.top_left_corner[1] += (self.current_roi_width - new_roi_width) // 2

        # Clipping these values to avoid going out of the original image:
        self.top_left_corner[0] = np.clip(a=self.top_left_corner[0],
                                          a_min=0,
                                          a_max=self.original_image.shape[0] - new_roi_height)

        self.top_left_corner[1] = np.clip(a=self.top_left_corner[1],
                                          a_min=0,
                                          a_max=self.original_image.shape[1] - new_roi_width)

        # Updating the frame:
        self.current_roi_height = new_roi_height
        self.current_roi_width = new_roi_width

        self.update_frame()

    def get_roi(self, display_shape):

        image = self.original_image.copy()
        bottom_right_corner = [self.top_left_corner[0] + self.current_roi_height,
                               self.top_left_corner[1] + self.current_roi_width]

        cv2.rectangle(image, self.top_left_corner[::-1], bottom_right_corner[::-1], (0, 0, 255), 10)

        image = cv2.resize(image, display_shape)
        return image

    def perform_action(self, action_type, action_details):

        if action_type == "Move":
            self.zoom_log = 0
            xc, yc = action_details[:2]
            xc *= self.current_roi_width
            yc *= self.current_roi_height

            xc, yc = int(xc), int(yc)

            if not self.move_log == [0, 0]:
                self.move(xc - self.move_log[0], self.move_log[1] - yc)

            self.move_log = [xc, yc]

        elif action_type == "Zoom":
            self.move_log = [0, 0]
            d = action_details
            if not self.zoom_log == 0:
                self.zoom(factor=self.zoom_log / d)

            self.zoom_log = d

        else:
            self.move_log = [0, 0]
            self.zoom_log = 0


def main():
    test_image_path = "test_image.jpg"
    test_image = cv2.imread(test_image_path)

    shape = (540, 960)
    image = Image(frame_shape=shape)
    image.update_image(test_image)

    STEP_SIZE = 15

    key_binds = {ord("e"): ("zoom", 0.95),
                 ord("r"): ("zoom", 1.05),
                 ord("w"): ("move", 0, -STEP_SIZE),
                 ord("a"): ("move", -STEP_SIZE, 0),
                 ord("s"): ("move", 0, STEP_SIZE),
                 ord("d"): ("move", STEP_SIZE, 0)}

    while True:

        frame = image.get_frame()
        cv2.imshow('frame', frame)

        roi = image.get_roi((576, 324))
        cv2.imshow("The original image + roi", roi)

        press = cv2.waitKey(1)

        # Press 'q' to exit the program.
        if press == ord("q"):
            break

        elif press in key_binds:
            action = getattr(image, key_binds[press][0])
            action(*key_binds[press][1:])

    cv2.destroyAllWindows()


# Testing the functionality:
# Keybindings: "e" to zoom in, "r" to zoom out , w_a_s_d to move.

if __name__ == "__main__":
    main()
