import autopy
import cv2
import mediapipe as mp
import numpy
from config import logger

from classifier.GestureClassifier import GestureClassifier
from mongo_api import save_landmarks_to_db
from old_version.save_gesture_utils import save_gesture_to_csv
from utils.CVFpsCalc import CvFpsCalc
from utils.utils import detect_hand_landmarks, write_image_to_file, convert_pad_point_to_screen_point

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

SCREEN_WIDTH, SCREEN_HEIGHT = (int(x) for x in autopy.screen.size())
SCALE_FACTOR = 0.5
PAD_WIDTH = SCREEN_WIDTH * SCALE_FACTOR
PAD_HEIGHT = SCREEN_WIDTH * SCALE_FACTOR
WRITE_TO_MONGO = False
SAVE_ONE_GESTURE = False
SAVE_GESTURES = False
GESTURE_TO_SAVE = None
SHAKE_CURSOR_RANGE = 5

gestures_classifier = GestureClassifier()


def draw_pad(frame: numpy.ndarray,
             shift_x: int = None,
             shift_y: int = None,
             width: int = PAD_WIDTH,
             height: int = PAD_HEIGHT):
    image_height, image_width, _ = frame.shape
    shift_x = (image_width - width) // 2 if shift_x is None else shift_x
    shift_y = (image_height - height) // 2 if shift_y is None else shift_y
    return cv2.rectangle(frame, (shift_x, shift_y), (shift_x + width, shift_y + height),
                         color=(255, 0, 255))


def process_data_from_camera(frame: numpy.ndarray, detect_landmarks: bool = False) -> (numpy.ndarray,):
    """
    Reads frame from camera and/or draws hand landmarks 
    :param frame:
    :rtype: (numpy.ndarray, tuple)
    :param detect_landmarks: is it necessary to detect and draw hand landmarks?
    :return: new frame
    """

    frame = cv2.flip(frame, 1)
    hand_landmarks = None

    if detect_landmarks:
        frame, results = detect_hand_landmarks(frame, hands_video)
        if WRITE_TO_MONGO:
            save_landmarks_to_db(results)
        if results and results.multi_hand_landmarks:
            hand_landmarks = next(iter(results.multi_hand_landmarks))

    return frame, hand_landmarks


def save_one_gesture(key: int, fingers_coords):
    if key == ord('1'):
        save_gesture_to_csv(1, fingers_coords)
        logger.info("Saved '1' gesture")
    elif key == ord('2'):
        save_gesture_to_csv(2, fingers_coords)
        logger.info("Saved '2' gesture")
    elif key == ord('3'):
        save_gesture_to_csv(3, fingers_coords)
        logger.info("Saved '3' gesture")


def check_save_gestures_flag(key: int):
    global GESTURE_TO_SAVE
    if key in (ord('1'), ord('2'), ord('3')):
        GESTURE_TO_SAVE = key - ord('1') + 1
        logger.info(f"Turned '{GESTURE_TO_SAVE}' flag for gesture")
    elif key == ord('s'):
        GESTURE_TO_SAVE = None
        logger.info(f"Turned off flag for gesture")


def get_finger_coords(hand_landmarks, shape):
    finger_tip_coords = None
    if hand_landmarks:
        image_height, image_width, _ = shape
        finger_tip_coords = (
            int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
        )
    return finger_tip_coords


def pre_process_landmark(landmarks):
    landmarks_list = []
    base_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    base_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    for i, landmark in enumerate(landmarks.landmark):
        landmarks_list.append(landmark.x - base_x)
        landmarks_list.append(landmark.y - base_y)
    return landmarks_list


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[0] - point2[0]) ** 2) ** 0.5


def place_mouse(hand_landmarks, frame, prev_screen_point):
    finger_coords = get_finger_coords(hand_landmarks, frame.shape)
    screen_point = None

    if finger_coords:
        frame = cv2.circle(frame, center=finger_coords, radius=15, color=(0, 255, 255), thickness=-1)

        pad_coords = (finger_coords[0] - shift_x, finger_coords[1] - shift_y)
        screen_point = convert_pad_point_to_screen_point(pad_coords,
                                                         pad_width=pad_width,
                                                         pad_height=pad_height,
                                                         screen_width=SCREEN_WIDTH,
                                                         screen_height=SCREEN_HEIGHT)
        if prev_screen_point and distance(prev_screen_point, screen_point) > SHAKE_CURSOR_RANGE:
            try:
                autopy.mouse.move(*screen_point)
            except ValueError as ignored:
                pass
    return frame, screen_point if screen_point else prev_screen_point


if __name__ == '__main__':
    hands_video = mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=1,
                                 min_detection_confidence=0.7,
                                 min_tracking_confidence=0.4)

    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 960)
    camera_video.set(4, 960)
    cv2.namedWindow('Hands Landmarks Detection', cv2.WINDOW_NORMAL)

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    scale_factor = SCALE_FACTOR
    shift_x = 140
    shift_y = 110
    logger.info('Application initialized')

    prev_click = None
    prev_screen_point = None

    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            logger.error('not opened')
            continue

        frame, hand_landmarks = process_data_from_camera(frame, detect_landmarks=True)

        # ==== function ====

        fps = cvFpsCalc.get()
        pad_width = int(SCREEN_WIDTH * scale_factor)
        pad_height = int(SCREEN_HEIGHT * scale_factor)
        frame = draw_pad(frame, width=pad_width, height=pad_height, shift_x=shift_x, shift_y=shift_y)

        cv2.putText(frame, 'FPS: {}'.format(fps),
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        if SAVE_GESTURES:
            cv2.putText(frame, f'GESTURE_TO_SAVE = {GESTURE_TO_SAVE}',
                        (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

        if GESTURE_TO_SAVE:
            save_gesture_to_csv(GESTURE_TO_SAVE, hand_landmarks)

        frame, prev_screen_point = place_mouse(hand_landmarks, frame, prev_screen_point)

        if hand_landmarks:
            landmarks_to_classify = pre_process_landmark(hand_landmarks)

            gesture_num = gestures_classifier(landmarks_to_classify) + 1
            if gesture_num != prev_click:
                if prev_click == 2:
                    autopy.mouse.toggle(button=autopy.mouse.Button.LEFT, down=False)
                elif prev_click == 3:
                    autopy.mouse.toggle(button=autopy.mouse.Button.RIGHT, down=False)

            if gesture_num == 2:
                prev_click = 2
                autopy.mouse.toggle(button=autopy.mouse.Button.LEFT, down=True)
            elif gesture_num == 3:
                prev_click = 3
                autopy.mouse.toggle(button=autopy.mouse.Button.RIGHT, down=True)
            cv2.putText(frame, f'GESTURE_NUM = {gesture_num}',
                        (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
        else:
            if prev_click == 2:
                autopy.mouse.toggle(button=autopy.mouse.Button.LEFT, down=False)
            elif prev_click == 3:
                autopy.mouse.toggle(button=autopy.mouse.Button.RIGHT, down=False)
            prev_click = None

        # ===== function end =====

        cv2.imshow('Hands Landmarks Detection', frame)

        # ==== parsing function ====

        # ======= default start =====
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            logger.info('STOP application')
            break
        elif k == ord('s'):
            logger.info('saved image')
            write_image_to_file(frame, dir='../saved_images')
        # ======= default end =====
        # ======= class start =====
        elif k == ord(','):
            scale_factor -= 0.05
            logger.info(f'scale_factor decreased to {scale_factor}')
        elif k == ord('.'):
            scale_factor += 0.05
            logger.info(f'scale_factor increased to {scale_factor}')
        elif k == ord('j'):
            shift_x -= 10
            logger.info(f'shift_x decreased to {shift_x}')
        elif k == ord(';'):
            shift_x += 10
            logger.info(f'shift_x increased to {shift_x}')
        elif k == ord('k'):
            shift_y -= 10
            logger.info(f'shift_y decreased to {shift_y}')
        elif k == ord('l'):
            shift_y += 10
            logger.info(f'shift_y increased to {shift_y}')
        # ======= class end =====
        # ======= train start =====
        elif SAVE_ONE_GESTURE:
            save_one_gesture(k, hand_landmarks)
        elif SAVE_GESTURES:
            check_save_gestures_flag(k)
        # ======= train end =====

    camera_video.release()
    cv2.destroyAllWindows()
