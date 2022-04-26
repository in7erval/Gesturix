CSV_FILENAME = "gestures.csv"

# TITLE = "gesture_num,1_x,1_y,2_x,2_y,3_x,3_y,4_x,4_y,5_x,5_y,6_x,6_y,7_x,7_y,8_x,8_y,9_x,9_y,10_x,10_y,11_x,11_y," \
#         "12_x,12_y,13_x,13_y,14_x,14_y,15_x,15_y,16_x,16_y,17_x,17_y,18_x,18_y,19_x,19_y,20_x,20_y "
TITLE = "gesture_num," + ",".join([f"{i}_x,{i}_y" for i in range(0, 21)])


def create_coordinates_str(hand_landmarks):
    coordinates_str = ""
    if hand_landmarks:
        for i in range(0, 21):
            landmark = hand_landmarks.landmark[i]
            coordinates_str += f"{landmark.x},{landmark.y},"
        return coordinates_str[:-1]
    return None


def save_gesture_to_csv(gesture_num, hand_landmarks):
    with open(CSV_FILENAME, 'a+') as f:
        coords_str = create_coordinates_str(hand_landmarks)
        if coords_str:
            f.write(f"{gesture_num},{coords_str}\n")
