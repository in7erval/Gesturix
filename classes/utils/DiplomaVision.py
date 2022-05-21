import logging
from enum import Enum
from typing import Dict, List

import cv2
import numpy as np

from classes.AppRunInterface import AppRunInterface

ALL_GESTURES_INFO_COLOR = (229, 43, 80)
CURRENT_GESTURE_INFO_COLOR = (80, 43, 229)
HELP_INFO_COLOR = (74, 148, 68)
DIGIT_KEYS = (ord('1'), ord('2'), ord('3'),
              ord('4'), ord('5'), ord('6'),
              ord('7'), ord('8'), ord('9'),
              ord('0'))


def load_gestures_from_csv(where_to_load: dict, fname, delim=','):
    try:
        with open(fname, 'r') as f:
            for line in f.readlines():
                gesture_num = line.split(delim, 1)[0]
                if gesture_num.isdigit():
                    gesture_num = int(gesture_num)
                if gesture_num in where_to_load.keys():
                    where_to_load[gesture_num] += 1
                else:
                    where_to_load[gesture_num] = 1
    except FileNotFoundError:
        logging.getLogger().warning(f'{fname} not found, skip loading gestures.')


class DiplomaVision(AppRunInterface):
    class Mode(Enum):
        CONTINUOUS = 1,
        SINGLE = 2,
        SEQUENTIAL = 3

    def __init__(self,
                 hands,
                 image_name,
                 filename: str = 'data/gestures_test.csv',
                 filename_seq: str = 'data/gestures_test_seq.csv',
                 ):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.hands = hands
        self.image = cv2.imread(image_name)
        self.gesture_to_save = None
        self.hand_landmarks = None
        self.saved_gestures_dict: Dict = {}
        self.gesture_sequence: List[List[float]] = []
        self.filename = filename
        self.filename_seq = filename_seq
        self.mode = DiplomaVision.Mode.SINGLE
        self.load_gestures()

    def __call__(self, frame, hand_landmarks):
        self.hand_landmarks = hand_landmarks

        self.put_text(frame)

        return frame

    def put_text(self, frame: np.ndarray):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (frame_width, 50), (200, 200, 200), thickness=-1)
        cv2.putText(frame, f'{self.create_gestures_string()}', (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, ALL_GESTURES_INFO_COLOR, 2)
        cv2.putText(frame, f"Gesture = '{self.gesture_to_save}' {self.mode}",
                    (10, 45), cv2.FONT_HERSHEY_PLAIN, 1.25, CURRENT_GESTURE_INFO_COLOR, 2)
        cv2.rectangle(frame, (0, frame_height - 70), (frame_width, frame_height), (200, 200, 200), thickness=-1)
        cv2.putText(frame, "To enable sequential gesture saving press 's', and then any digit key for gesture.",
                    (10, frame_height - 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, HELP_INFO_COLOR, 2)
        cv2.putText(frame, "To save current gesture press any digit key.",
                    (10, frame_height - 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, HELP_INFO_COLOR, 2)
        cv2.putText(frame, "To continuous gesture saving press 'c', and then any digit key for gesture (for fast "
                           "generating static gestures)",
                    (10, frame_height - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, HELP_INFO_COLOR, 2)

    def parse_keyboard(self, key):
        if key in DIGIT_KEYS:
            self.gesture_to_save = key - ord('0')
        elif key == ord('c'):
            self.mode = DiplomaVision.Mode.CONTINUOUS \
                if self.mode is not DiplomaVision.Mode.CONTINUOUS else DiplomaVision.Mode.SINGLE
        elif key == ord('s'):
            self.gesture_to_save = None
            self.mode = DiplomaVision.Mode.SEQUENTIAL \
                if self.mode is not DiplomaVision.Mode.SEQUENTIAL else DiplomaVision.Mode.SINGLE

    def create_gestures_string(self):
        s = ""
        for k, v in self.saved_gestures_dict.items():
            s += f"['{k}': {v}] "
        return s

    def load_gestures(self):
        self.saved_gestures_dict = {}
        load_gestures_from_csv(self.saved_gestures_dict, self.filename)
        load_gestures_from_csv(self.saved_gestures_dict, self.filename_seq, delim=' | ')
        self.logger.info('Gestures dict loaded')
        self.logger.debug(f'{self.saved_gestures_dict}')
