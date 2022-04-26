import argparse
import logging

import cv2
import mediapipe as mp
import numpy

from classes.AppRunInterface import AppRunInterface
from classes.GestureClassification import GestureClassification
from classes.GesturesTraining import GesturesTraining
from utils import detect_hand_landmarks


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_mode', '-lm', action='store_true', default=False,
                        help='Mode to learn new gestures')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--width', help='screen capture width', type=int, default=960)
    parser.add_argument('--height', help='screen capture height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true', default=False)
    parser.add_argument('--min_detection_confidence',
                        help='MIN_DETECTION_CONFIDENCE = [0 ... 1]',
                        type=float,
                        default=0.7)
    parser.add_argument('--min_tracking_confidence',
                        help='MIN_TRACKING_CONFIDENCE = [0 ... 1]',
                        type=float,
                        default=0.5)

    return parser.parse_args()


class App:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.args = get_args()
        self.hands = self.get_hands()
        self.camera = self.get_camera()

        if self.args.learning_mode:
            self.logger.info("Learning mode is activated!")
            self.run_class: AppRunInterface = GesturesTraining(self.hands,
                                                               self.camera)
        else:
            self.logger.info("Classification mode is activated!")
            self.run_class: AppRunInterface = GestureClassification(self.hands,
                                                                    self.camera)

    def get_hands(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=self.args.use_static_image_mode,
            max_num_hands=1,
            min_tracking_confidence=self.args.min_tracking_confidence,
            min_detection_confidence=self.args.min_detection_confidence
        )
        return self.hands

    def get_camera(self):
        self.camera = cv2.VideoCapture(self.args.device)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        return self.camera

    def start(self):
        cv2.namedWindow('Hands Landmarks Detection', cv2.WINDOW_NORMAL)

        while self.camera.isOpened():
            ok, frame = self.camera.read()
            if not ok:
                self.logger.error('not opened')
                continue

            frame, hand_landmarks = self.process_data_from_camera(frame, detect_landmarks=True)

            frame = self.run_class(frame, hand_landmarks)

            cv2.imshow('Hands Landmarks Detection', frame)

            key = cv2.waitKeyEx(1)

            self.run_class.parse_keyboard(key)

            if key == 27 or key == ord('q'):
                self.logger.info('STOP application')
                break

        self.camera.release()
        cv2.destroyAllWindows()

    def process_data_from_camera(self, frame: numpy.ndarray, detect_landmarks: bool = False) -> (numpy.ndarray,):
        frame = cv2.flip(frame, 1)
        hand_landmarks = None

        if detect_landmarks:
            frame, results = detect_hand_landmarks(frame, self.hands)
            if results and results.multi_hand_landmarks:
                hand_landmarks = next(iter(results.multi_hand_landmarks))

        return frame, hand_landmarks


if __name__ == '__main__':
    App().start()
