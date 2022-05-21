import logging

import autopy
import cv2
import mediapipe as mp
import numpy

from classes.AppRunInterface import AppRunInterface
from classes.DynamicBuffer import DynamicBuffer
from classifier.GestureClassifier import GestureClassifier
from utils.CVFpsCalc import CvFpsCalc
from utils.utils import landmarks_to_plain_list, relativize

SHAKE_CURSOR_RANGE = 5
FPS_COLOR = (229, 43, 80)
GESTURE_INFO_COLOR = (80, 43, 229)


def get_finger_coords(hand_landmarks, shape):
    finger_tip_coords = None
    if hand_landmarks:
        image_height, image_width, _ = shape
        finger_tip_coords = (
            int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * image_width),
            int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
        )
    return finger_tip_coords


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[0] - point2[0]) ** 2) ** 0.5


def pre_process_landmark(landmarks):
    landmarks_list = []
    base_x = landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x
    base_y = landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y
    for i, landmark in enumerate(landmarks.landmark):
        landmarks_list.append(landmark.x - base_x)
        landmarks_list.append(landmark.y - base_y)
    return landmarks_list


class GestureClassification(AppRunInterface):

    def __init__(self,
                 hands,
                 camera: cv2.VideoCapture,
                 scale_factor: float = 0.5):
        self.prev_screen_point = None
        self.finger_coords = None
        self.prev_click = None
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.hands = hands
        self.camera = camera
        self.scale_factor = scale_factor
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.screen_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.screen_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.shift_x = int(self.screen_width * (1 - scale_factor) // 2)
        self.shift_y = int(self.screen_height * (1 - scale_factor) // 2)
        self.pad_width = int(self.screen_width * self.scale_factor)
        self.pad_height = int(self.screen_height * self.scale_factor)
        self.gestures_classifier = GestureClassifier()
        self.dynamic_gestures_classifier = GestureClassifier(model_path='data/gestures_sequence_classifier.tflite')
        self.buffer = DynamicBuffer(buffer_size=self.dynamic_gestures_classifier.get_input_shape()[1])

    def __call__(self, frame, hand_landmarks):
        fps = self.cvFpsCalc.get()
        frame = self.draw_pad(frame)

        self.buffer.save(landmarks_to_plain_list(hand_landmarks))

        self.hand_landmarks = hand_landmarks
        cv2.putText(frame, 'FPS: {}'.format(fps),
                    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.25, FPS_COLOR, 2)

        self.finger_coords = get_finger_coords(hand_landmarks, frame.shape)

        if self.finger_coords:
            frame = cv2.circle(frame, center=self.finger_coords,
                               radius=15, color=(0, 255, 255), thickness=-1)

            self.place_mouse()

        if hand_landmarks:
            landmarks_to_classify = pre_process_landmark(hand_landmarks)

            if self.buffer.is_full():
                dynamic_gesture_num, ver = self.dynamic_gestures_classifier(relativize(self.buffer.get()))
                if dynamic_gesture_num != 2:
                    self.logger.info(f'Dynamic gesture found! Gesture_num: {dynamic_gesture_num} {ver}')
            gesture_num, _ = self.gestures_classifier(landmarks_to_classify)
            gesture_num += 1
            if gesture_num != self.prev_click:
                if self.prev_click == 2:
                    autopy.mouse.toggle(button=autopy.mouse.Button.LEFT, down=False)
                    self.logger.debug('Left button toggle up')
                elif self.prev_click == 3:
                    autopy.mouse.toggle(button=autopy.mouse.Button.RIGHT, down=False)
                    self.logger.debug('Right button toggle up')

                if gesture_num == 2:
                    self.prev_click = 2
                    autopy.mouse.toggle(button=autopy.mouse.Button.LEFT, down=True)
                    self.logger.debug('Left button toggle down')
                elif gesture_num == 3:
                    self.prev_click = 3
                    autopy.mouse.toggle(button=autopy.mouse.Button.RIGHT, down=True)
                    self.logger.debug('Right button toggle down')
            cv2.putText(frame, f'GESTURE_NUM = {gesture_num}',
                        (10, 45), cv2.FONT_HERSHEY_PLAIN, 1.25, GESTURE_INFO_COLOR, 2)
            self.prev_click = gesture_num
        else:
            if self.prev_click == 2:
                autopy.mouse.toggle(button=autopy.mouse.Button.LEFT, down=False)
            elif self.prev_click == 3:
                autopy.mouse.toggle(button=autopy.mouse.Button.RIGHT, down=False)
            self.prev_click = None
        return frame

    def parse_keyboard(self, key):
        if key == ord(','):
            self.scale_factor -= 0.05
            self.pad_width = int(self.screen_width * self.scale_factor)
            self.pad_height = int(self.screen_height * self.scale_factor)
            self.logger.info(f'scale_factor decreased to {self.scale_factor}')
        elif key == ord('.'):
            self.scale_factor += 0.05
            self.pad_width = int(self.screen_width * self.scale_factor)
            self.pad_height = int(self.screen_height * self.scale_factor)
            self.logger.info(f'scale_factor increased to {self.scale_factor}')
        elif key == ord('j'):
            self.shift_x -= 10
            self.logger.info(f'shift_x decreased to {self.shift_x}')
        elif key == ord(';'):
            self.shift_x += 10
            self.logger.info(f'shift_x increased to {self.shift_x}')
        elif key == ord('k'):
            self.shift_y -= 10
            self.logger.info(f'shift_y decreased to {self.shift_y}')
        elif key == ord('l'):
            self.shift_y += 10
            self.logger.info(f'shift_y increased to {self.shift_y}')

    def draw_pad(self, frame: numpy.ndarray):
        image_height, image_width, _ = frame.shape
        return cv2.rectangle(frame, (self.shift_x, self.shift_y),
                             (self.shift_x + self.pad_width, self.shift_y + self.pad_height),
                             color=(255, 0, 255))

    def place_mouse(self):
        pad_coords = (self.finger_coords[0] - self.shift_x, self.finger_coords[1] - self.shift_y)
        screen_point = self.convert_pad_point_to_screen_point(pad_coords)
        if self.prev_screen_point and distance(self.prev_screen_point, screen_point) > SHAKE_CURSOR_RANGE:
            try:
                autopy.mouse.move(*screen_point)
            except ValueError:
                pass
        self.prev_screen_point = screen_point

    def convert_pad_point_to_screen_point(self, pad_point: tuple) -> tuple:
        scale_x = self.screen_width / self.pad_width
        scale_y = self.screen_height / self.pad_height
        return pad_point[0] * scale_x, pad_point[1] * scale_y
