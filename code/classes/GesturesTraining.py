import logging
from enum import Enum
from typing import Dict, List

import cv2
import numpy as np

from classes.AppRunInterface import AppRunInterface
from utils.utils import landmarks_to_plain_list

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
    with open(fname, 'a+') as f:
        coords_str = ",".join(map(str, landmarks_to_plain_list(hand_landmarks)))
        if coords_str:
            f.write(f"{gesture_num},{coords_str}\n")


def save_gesture_sequence_to_csv(fname, gesture_num, gesture_sequence):
    with open(fname, 'a+') as f:
        coords_str = ";".join(map(str, gesture_sequence))
        if coords_str:
            f.write(f"{gesture_num} | {coords_str}\n")


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


class GesturesTraining(AppRunInterface):
    class Mode(Enum):
        CONTINUOUS = 1,
        SINGLE = 2,
        SEQUENTIAL = 3

    def __init__(self,
                 hands,
                 camera,
                 filename: str = 'data/gestures_test.csv',
                 filename_seq: str = 'data/gestures_test_seq.csv',
                 mode: Mode = Mode.SINGLE):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.hands = hands
        self.camera = camera
        self.mode = mode
        self.gesture_to_save = None
        self.filename = filename
        self.filename_seq = filename_seq
        self.hand_landmarks = None
        self.saved_gestures_dict: Dict = None
        self.gesture_sequence: List[List[float]] = []
        self.load_gestures()

    def __call__(self, frame, hand_landmarks):
        self.hand_landmarks = hand_landmarks

        self.put_text(frame)

        if self.mode is not GesturesTraining.Mode.SINGLE:
            self.save_gesture()
        return frame

    def put_text(self, frame: np.ndarray):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (frame_width, 50), (200, 200, 200), thickness=-1)
        gestures_str = self.create_gestures_string()
        if len(gestures_str) == 0:
            cv2.putText(frame, 'No data available', (10, 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.25, ALL_GESTURES_INFO_COLOR, 2)
        else:
            cv2.putText(frame, f'{gestures_str}', (10, 20),
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
            if self.mode is GesturesTraining.Mode.SEQUENTIAL and self.gesture_to_save is not None:
                self.action_when_change_sequential_mode()
            self.gesture_to_save = key - ord('0')
            if self.mode is GesturesTraining.Mode.CONTINUOUS:
                self.logger.info(f"Begin to save '{self.gesture_to_save}' gesture. Mode = {self.mode}")
            elif self.mode is GesturesTraining.Mode.SEQUENTIAL:
                self.logger.info(f"Begin to save '{self.gesture_to_save}' gesture. Mode = {self.mode}")
            else:
                self.save_gesture()
                self.logger.info(f"Saved '{self.gesture_to_save}' gesture")
        elif key == ord('c'):
            self.gesture_to_save = None
            self.action_when_change_sequential_mode()
            self.mode = GesturesTraining.Mode.CONTINUOUS \
                if self.mode is not GesturesTraining.Mode.CONTINUOUS else GesturesTraining.Mode.SINGLE
            off_on = 'on' if self.mode is GesturesTraining.Mode.CONTINUOUS else 'off'
            self.logger.info(f'Turned {off_on} flag for continuous gesture save')
        elif key == ord('s'):
            self.gesture_to_save = None
            self.action_when_change_sequential_mode()
            self.mode = GesturesTraining.Mode.SEQUENTIAL \
                if self.mode is not GesturesTraining.Mode.SEQUENTIAL else GesturesTraining.Mode.SINGLE
            off_on = 'on' if self.mode is GesturesTraining.Mode.SEQUENTIAL else 'off'
            self.logger.info(f'Turned {off_on} flag for sequential gesture save')

    def save_gesture(self):
        if self.gesture_to_save is not None and self.hand_landmarks:
            if self.mode is GesturesTraining.Mode.CONTINUOUS:
                save_gesture_to_csv(self.filename,
                                    self.gesture_to_save,
                                    self.hand_landmarks)
                if self.gesture_to_save not in self.saved_gestures_dict.keys():
                    self.saved_gestures_dict[self.gesture_to_save] = 1
                else:
                    self.saved_gestures_dict[self.gesture_to_save] += 1
            elif self.mode is GesturesTraining.Mode.SEQUENTIAL:
                self.gesture_sequence.append(landmarks_to_plain_list(self.hand_landmarks))

    def load_gestures(self):
        self.saved_gestures_dict = {}
        load_gestures_from_csv(self.saved_gestures_dict, self.filename)
        load_gestures_from_csv(self.saved_gestures_dict, self.filename_seq, delim=';')
        self.logger.info('Gestures dict loaded')
        self.logger.debug(f'{self.saved_gestures_dict}')

    def create_gestures_string(self):
        s = ""
        for k, v in self.saved_gestures_dict.items():
            s += f"['{k}': {v}] "
        return s

    def action_when_change_sequential_mode(self):
        """
            Перед сменой режима и номера жеста сохраняем сохранённую последовательность,
            если она вообще есть
        """

        if self.mode is GesturesTraining.Mode.SEQUENTIAL and self.gesture_to_save:
            gesture_num = str(self.gesture_to_save) + 's'
            save_gesture_sequence_to_csv(self.filename_seq,
                                         gesture_num,
                                         self.gesture_sequence)

            if gesture_num not in self.saved_gestures_dict.keys():
                self.saved_gestures_dict[gesture_num] = 1
            else:
                self.saved_gestures_dict[gesture_num] += 1
            self.gesture_sequence = []
