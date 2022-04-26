import logging
from enum import Enum
from typing import Dict

import cv2
import numpy as np

from classes.AppRunInterface import AppRunInterface

DIGIT_KEYS = (ord('1'), ord('2'), ord('3'),
              ord('4'), ord('5'), ord('6'),
              ord('7'), ord('8'), ord('9'),
              ord('0'))
ALL_GESTURES_INFO_COLOR = (229, 43, 80)
CURRENT_GESTURE_INFO_COLOR = (80, 43, 229)
HELP_INFO_COLOR = (74, 148, 68)


def hex2rgb(hex_str: str) -> tuple:
    """
    Converts HEX format to RGB
    :param hex_str:
    :return: tuple(r, g, b)
    """
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def save_gesture_to_csv(fname, gesture_num, hand_landmarks):
    def create_coordinates_str(hl):
        coordinates_str = ""
        if hl:
            for i in range(0, 21):
                landmark = hl.landmark[i]
                coordinates_str += f"{landmark.x},{landmark.y},"
            return coordinates_str[:-1]
        return None

    with open(fname, 'a+') as f:
        coords_str = create_coordinates_str(hand_landmarks)
        if coords_str:
            f.write(f"{gesture_num},{coords_str}\n")


def load_gestures_from_csv(fname):
    d = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            gesture_num = int(line.split(',', 1)[0])
            if gesture_num in d.keys():
                d[gesture_num] += 1
            else:
                d[gesture_num] = 1
    return d


class GesturesTraining(AppRunInterface):
    class Mode(Enum):
        CONTINUOUS = 1,
        SINGLE = 2

    def __init__(self,
                 hands,
                 camera,
                 filename: str = 'gestures_test.csv',
                 mode: Mode = Mode.SINGLE):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.hands = hands
        self.camera = camera
        self.mode = mode
        self.gesture_to_save = None
        self.filename = filename
        self.hand_landmarks = None
        self.saved_gestures_dict: Dict[int, int] = None
        self.load_gestures()

    def __call__(self, frame, hand_landmarks):
        self.hand_landmarks = hand_landmarks

        self.put_text(frame)

        if self.mode == GesturesTraining.Mode.CONTINUOUS:
            self.save_gesture()
        return frame

    def put_text(self, frame: np.ndarray):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (frame_width, 50), (200, 200, 200), thickness=-1)
        cv2.putText(frame, f'{self.create_gestures_string()}', (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, ALL_GESTURES_INFO_COLOR, 2)
        cv2.putText(frame, f"Gesture = '{self.gesture_to_save}' {self.mode}",
                    (10, 45), cv2.FONT_HERSHEY_PLAIN, 1.25, CURRENT_GESTURE_INFO_COLOR, 2)
        cv2.rectangle(frame, (0, frame_height - 50), (frame_width, frame_height), (200, 200, 200), thickness=-1)
        cv2.putText(frame, "To save current gesture press any digit key.",
                    (10, frame_height - 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, HELP_INFO_COLOR, 2)
        cv2.putText(frame, "To continuous gesture saving press 's', and then any digit key for gesture",
                    (10, frame_height - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, HELP_INFO_COLOR, 2)

    def parse_keyboard(self, key):
        if key in DIGIT_KEYS:
            self.gesture_to_save = key - ord('0')
            if self.mode == GesturesTraining.Mode.CONTINUOUS:
                self.logger.info(f"Begin to save '{self.gesture_to_save}' gesture")
            else:
                self.save_gesture()
                self.logger.info(f"Saved '{self.gesture_to_save}' gesture")
        elif key == ord('s'):
            self.gesture_to_save = None
            off_on = 'on' if self.mode == GesturesTraining.Mode.SINGLE else 'off'
            self.mode = GesturesTraining.Mode.CONTINUOUS \
                if self.mode == GesturesTraining.Mode.SINGLE else GesturesTraining.Mode.SINGLE
            self.logger.info(f'Turned {off_on} flag for continuous gesture save')

    def save_gesture(self):
        if self.gesture_to_save is not None:
            save_gesture_to_csv(self.filename,
                                self.gesture_to_save,
                                self.hand_landmarks)
            if self.hand_landmarks:
                if self.gesture_to_save not in self.saved_gestures_dict.keys():
                    self.saved_gestures_dict[self.gesture_to_save] = 1
                else:
                    self.saved_gestures_dict[self.gesture_to_save] += 1

    def load_gestures(self):
        self.saved_gestures_dict = load_gestures_from_csv(self.filename)
        self.logger.info('Gestures dict loaded')
        self.logger.debug(f'{self.saved_gestures_dict}')

    def create_gestures_string(self):
        s = ""
        for k, v in self.saved_gestures_dict.items():
            s += f"['{k}': {v}] "
        return s
