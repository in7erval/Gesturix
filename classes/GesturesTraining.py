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
COLORS = {
    "Абрикосовый": (251, 206, 177),  # Абрикосовый
    "Абрикосовый Крайола": (253, 217, 181),  # Абрикосовый Крайола
    "Агатовый серый": (181, 184, 177),  # Агатовый серый
    "Аквамариновый": (127, 255, 212),  # Аквамариновый
    "Аквамариновый Крайола": (120, 219, 226),  # Аквамариновый Крайола
    "Ализариновый красный": (227, 38, 54),  # Ализариновый красный
    "Алый": (255, 36, 0),  # Алый
    "Амарантово-пурпурный": (171, 39, 79),
    "Амарантово-розовый": (241, 156, 187),
    "Амарантовый": (229, 43, 80),
    "Амарантовый глубоко-пурпурный": (159, 43, 104),
    "Амарантовый маджента": (237, 60, 202),
    "Амарантовый светло-вишневый": (205, 38, 130),
    "Американский розовый": (255, 3, 62),
    "Аметистовый": (153, 102, 204),
    "Античная латунь": (205, 149, 117),
    "Антрацитово-серый": (41, 49, 51),
    "Антрацитовый": (70, 68, 81),
    "Арлекин": (68, 148, 74),
    "Аспидно-синий": (106, 90, 205),
    "Бабушкины яблоки": (168, 228, 160)
    # ...
}

PALETTES = {
    "Розовый и изюм": ['e52165', '0d1137'],
    "Красный, морской пены, нефрита и фиалки": ['d72631', 'a2d5c6',
                                                '077b8a', '5c3c92'],
    "Желтый, пурпурный, голубой, черный": ['e2d810', 'd9138a',
                                           '12a4d9', '322e2f'],
    "Горчица и черный": ['f3ca20', '000000'],
    "Пурпурный, золотарник, бирюза и кирпич": ['cf1578', '38d21d',
                                               '039fbe', 'b20238'],
    "Оттенки розового и коричневого": ['e75874', 'be1558',
                                       'fbcbc9', '322514'],
    "Золото, уголь и серый": ['ef9d10f', '3b4d61', '6b7b8c'],
    "Военно-морской флот, миндаль, красно-оранжевый и манго": ['1e3d59', 'f5f0e1',
                                                               'ff6e40', 'ffc13b'],
    "Загар, глубокий бирюзовый и черный": ['ecc19c', '1e847f', '000000'],
    "Военно-морской флот, охра, сожженная сиена и светло-серый": ['26495c', 'c4a35a',
                                                                  'c66b3d', 'e5e5dc'],
    "Сиреневый, сапфировый и пудрово-серый": ['d9a6b3', '1868ae', 'c6d7eb'],
    "Синий, бордовый и индиго": ['408ec6', '7a2048', '1e2761'],
    "Малина и оттенки синего": ['8a307f', '79a7d3', '6883bc'],
    "Глубокий сосново-зеленый, оранжевый и светло-персиковый": ['1d3c45', 'd2601a', 'fff1e1'],
    "Морская пена, лосось и флот": ['aed6dc', 'ff9a8d', '4a536b'],
    "Руж, зеленый и пурпурный": ['da68a0', '77c593', 'ed3572'],
    "Чирок, коралл, бирюза и серый": ['316879', 'f47a60', '7fe7dc', 'ced7d8'],
    "Фуксия, сепия, ярко-розовый и темно-фиолетовый": ['d902ee', 'ffd79d', 'f162ff', '320d3e'],
    "Светло-розовый, шалфей, голубой и виноград": ['ffcce7', 'daf2dc', '81b7d2', '4d5198'],
    "Бежевый, черно-коричневый и желто-коричневый": ['ddc3a5', '201e20', 'e0a96d'],
    "Сепия, чирок, беж и шалфей": ['edca82', '097770', 'e0cdbe', 'a9c0a6'],
    "Желто-зеленый, оливковый и лесной зеленый": ['e1dd72', 'a8c66c', '1b6535'],
    "Фуксия, желтый и пурпурный": ['d13ca4', 'ffea04', 'fe3a9e'],
    "Горчица, шалфей и зеленый лес": ['e3b448', 'cbd18f', '3a6b35'],
    "Бежевый, шифер и хаки": ['f6ead4', 'a2a595', 'b4a284'],
    "Бирюзовый и фиолетовый": ['79cbb8', '500472'],
    "Светло-розовый, зеленый и морской пены": ['f5beb4', '9bc472', 'cbf6db'],
    "Алый, светло-оливковый и светло-бирюзовый": ['b85042', 'e7e8d1', 'a7beae'],
    "Красный, желтый, голубой и ярко-фиолетовый": ['d71b3b', 'e8d71e', '16acea', '4203c9'],
    "Оливковое, бежевое и коричневое": ['829079', 'ede6b9', 'b9925e'],
    "Оттенки синего и зеленого": ['1fbfb8', '05716c', '1978a5', '031163'],
    "Бирюзовый, горчичный и черный": ['7fc3c0', 'cfb845', '141414'],
    "Персик, лосось и чирок": ['efb5a3', 'f57e7e', '315f72']
}


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
