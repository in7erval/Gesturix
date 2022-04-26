import os.path
import time
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def detect_hand_landmarks(image: np.array, hands: mp.solutions.hands.Hands) -> (np.array, NamedTuple):
    """
    Detect and draw hand landmarks in image
    :rtype: tuple
    :param image: Frame
    :param hands: object of mediapipe.solutions.hands.Hands
    :return: tuple of (output_image, results), where *output_image* is
    a copy of image with landmarks and *results* -- result of hands.process(image)
    (image converted to RGB)
    """

    mp_drawing_styles = mp.solutions.drawing_styles
    image_height, image_width, _ = image.shape
    output_image = image.copy()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # logger.debug(
            #     f'Index finger tip coordinate: ('
            #     f'{int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)}, '
            #     f'{int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)})'
            # )
            mp_drawing.draw_landmarks(image=output_image,
                                      landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                                      connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    # if display:
    #     plt.figure(figsize=[15, 15])
    #     plt.subplot(121)
    #     plt.imshow(image[:, :, ::-1])
    #     plt.title("Original image")
    #     plt.axis("off")
    #     plt.subplot(122)
    #     plt.imshow(output_image[:, :, ::-1])
    #     plt.title("Processed image")
    #     plt.axis('off')
    #     plt.show()
    # else:
    return output_image, results


def write_image_to_file(image, filename=None, dir=None):
    """
    Writes image to file, if directory is not created, creates directory
    :param filename: if None, uses image{time.strftime("%Y-%m-%d_%H:%M:%S")}.jpg'
    :param dir:
    :return: None
    """

    if filename is None:
        filename = f'image{time.strftime("%Y-%m-%d_%H:%M:%S")}.jpg'
    curr_dir = os.path.dirname(__file__)
    if dir is not None:
        dir = os.path.join(curr_dir, dir)
        if not os.path.exists(dir):
            os.mkdir(dir)
        filename = os.path.join(dir, filename)
    cv2.imwrite(filename, image)
    # logger.debug(f'{filename} saved')


def convert_pad_point_to_screen_point(pad_point: tuple, pad_width: int, pad_height: int,
                                      screen_width: int, screen_height: int) -> tuple:
    scale_x = screen_width / pad_width
    scale_y = screen_height / pad_height
    return pad_point[0] * scale_x, pad_point[1] * scale_y


def process_data_from_camera(hands, frame: numpy.ndarray, detect_landmarks: bool = False) -> (numpy.ndarray,):
    """
    Reads frame from camera and/or draws hand landmarks
    :param hands:
    :param frame:
    :rtype: (numpy.ndarray, tuple)
    :param detect_landmarks: is it necessary to detect and draw hand landmarks?
    :return: new frame
    """

    frame = cv2.flip(frame, 1)
    hand_landmarks = None

    if detect_landmarks:
        frame, results = detect_hand_landmarks(frame, hands)
        # if WRITE_TO_MONGO:
        #     save_landmarks_to_db(results)
        if results and results.multi_hand_landmarks:
            hand_landmarks = next(iter(results.multi_hand_landmarks))

    return frame, hand_landmarks