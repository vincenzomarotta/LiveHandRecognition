import cv2
import numpy as np
import mediapipe as mp
import argparse

from heuristic_model import heuristic_model
from utils.utility import *

import nn_model.nn_model_torch as nn_model


GESTURE = ['CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK', 'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK',
                'STOP', 'THREE', 'TWO']
HEURISTIC = 0
NN = 1

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(prog='Hand Gesture Classification')
    # parser.add_argument("t", type=int,
    #                     help="Type of classifier: Heuristic (0), Neural Network (1)",
    #                     choices=[0, 1])
    # parser.add_argument("w", type=int, help="Frame width")
    # parser.add_argument("ht", type=int, help="Frame height")

    # args = parser.parse_args()

    mode = 0

    # model_type = args.t
    # frame_width = args.w
    # frame_height = args.s
    model_type = 1
    frame_width = 600
    frame_height = 600

    heuristic_classifier = heuristic_model.HeuristicClassifier((frame_width, frame_height))
    nn_classifier = nn_model.GestureClassifierNN()
    nn_classifier.load("./nn_model/gesture_nn_model.pth")

    # Initialize hand class and store it into a variable
    mp_hands = mp.solutions.hands

    # Drawing functions and styles to draw landmarks on the frame
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(1)

    with mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            number, mode = select_mode(key, mode)

            # Capture frame by frame
            ret, frame = cap.read()

            # Mirror frame
            frame = cv2.flip(frame, 1)

            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)

            # Draw the hand annotations on the frame
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Create a list of landmark with frame coordinate
                    landmarks = pre_process_landmark(hand_landmarks, frame)

                    # HEURISTIC CLASSIFIER
                    if model_type == HEURISTIC:
                        gesture_code = heuristic_classifier.process(landmarks)

                    # NN CLASSIFIER
                    if model_type == NN:
                        norm_landmarks = normalize_wrt_keypoint(landmarks, landmarks[0], one_d=True)

                        # Eventually write to the dataset file
                        logging_csv(number, mode, norm_landmarks)
                        frame = draw_info(frame, mode, number)

                        gesture_code = nn_classifier.predict(norm_landmarks)

                    brec = bounding_rec(landmarks)
                    frame = draw_info_text(frame, brec, handedness, GESTURE[gesture_code])
                    print(f"Predicted gesture: {GESTURE[gesture_code]}")

                    cv2.imshow('Hand Image', frame)

        cap.release()
        cv2.destroyAllWindows()
