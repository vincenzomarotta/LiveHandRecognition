import numpy as np
import copy
from itertools import chain
import cv2
import math
import csv


def calculate_finger_angles(pure_landmarks):
    """
    Calcola gli angoli di piegamento usando le coordinate pure di MediaPipe.
    """
    angles = []
    joint_triplets = [
        (1, 2, 3), (2, 3, 4),  # Pollice
        (5, 6, 7), (6, 7, 8),  # Indice
        (9, 10, 11), (10, 11, 12),  # Medio
        (13, 14, 15), (14, 15, 16),  # Anulare
        (17, 18, 19), (18, 19, 20)  # Mignolo
    ]

    lm = np.array(pure_landmarks)

    for a, b, c in joint_triplets:
        v1 = lm[a] - lm[b]
        v2 = lm[c] - lm[b]

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle_norm = np.degrees(angle_rad) / 180.0
        angles.append(angle_norm)

    return angles


def pre_process_landmark(landmark_list, image):
    """
    Estrae X, Y (in pixel) e Z (relativo) dai landmark di MediaPipe.
    Restituisce una lista di 21 liste: [[x0, y0, z0], [x1, y1, z1], ...]
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmark_list.landmark):
        # Mappa X e Y ai pixel dell'immagine
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        # Mantiene Z così com'è (è già un valore relativo alla scala della mano)
        landmark_z = landmark.z

        # IMPORTANTE: Aggiungiamo tutte e 3 le coordinate!
        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point


def normalize_wrt_keypoint(landmarks, keypoint, one_d=False):
    tmp_landmarks = copy.deepcopy(landmarks)

    base_x = keypoint[0]
    base_y = keypoint[1]
    base_z = keypoint[2]

    for index, landmark_point in enumerate(tmp_landmarks):
        # 2. Sottraiamo il punto base anche all'asse Z
        tmp_landmarks[index][0] = tmp_landmarks[index][0] - base_x
        tmp_landmarks[index][1] = tmp_landmarks[index][1] - base_y
        tmp_landmarks[index][2] = tmp_landmarks[index][2] - base_z

        # Convert to one-dimensional list
    one_d_landmarks = list(chain.from_iterable(tmp_landmarks))

    # Take the maximum value from one-dimensional list
    max_value = max(list(map(abs, one_d_landmarks)))

    # Prevenzione divisione per zero (sicurezza)
    if max_value == 0:
        max_value = 1.0

    # Normalize the value
    normalize = lambda x: x / max_value

    if one_d:
        tmp_landmarks = list(map(normalize, one_d_landmarks))
    else:
        tmp_landmarks = [[normalize(coord) for coord in point] for point in tmp_landmarks]

    return tmp_landmarks


def bounding_rec(landmarks):
    landmark_array = np.empty((0, 2), int)
    for landmark in landmarks:
        landmark_point = [np.array((landmark[0], landmark[1]), dtype=np.int16)]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def draw_info_text(frame, brect, handedness, hand_sign_text):
    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    info_text = info_text + ': ' + hand_sign_text

    cv2.putText(frame, info_text, ((brect[0] + 5, brect[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_info(frame, mode, number):
    mode_string = ['Logging Key Point']
    if mode == 1:
        cv2.putText(frame, "MODE: " + mode_string[mode - 1], (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

        if 0 <= number <= 9:
            cv2.putText(frame, "NUM: " + str(number), (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48

    if key == 110:  #  n
        mode = 0
    if key == 107:  # k
        mode = 1

    return number, mode


def draw_hand(frame, landmarks):
    scaled_landmarks = []
    radius = 5
    for _, landmark in enumerate(landmarks):
        landmark_x = int(landmark[0])
        landmark_y = int(landmark[1])

        scaled_landmarks.append((landmark_x, landmark_y))
        cv2.circle(frame, (landmark_x, landmark_y), radius, (0, 0, 255), 5)

    cv2.line(frame, tuple(scaled_landmarks[0]), tuple(scaled_landmarks[1]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[1]), tuple(scaled_landmarks[2]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[2]), tuple(scaled_landmarks[3]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[3]), tuple(scaled_landmarks[4]), (0, 0, 255), 2)

    cv2.line(frame, tuple(scaled_landmarks[0]), tuple(scaled_landmarks[5]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[5]), tuple(scaled_landmarks[6]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[6]), tuple(scaled_landmarks[7]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[7]), tuple(scaled_landmarks[8]), (0, 0, 255), 2)

    cv2.line(frame, tuple(scaled_landmarks[5]), tuple(scaled_landmarks[9]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[9]), tuple(scaled_landmarks[10]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[10]), tuple(scaled_landmarks[11]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[11]), tuple(scaled_landmarks[12]), (0, 0, 255), 2)

    cv2.line(frame, tuple(scaled_landmarks[9]), tuple(scaled_landmarks[13]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[13]), tuple(scaled_landmarks[14]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[14]), tuple(scaled_landmarks[15]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[15]), tuple(scaled_landmarks[16]), (0, 0, 255), 2)

    cv2.line(frame, tuple(scaled_landmarks[0]), tuple(scaled_landmarks[17]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[13]), tuple(scaled_landmarks[17]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[17]), tuple(scaled_landmarks[18]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[18]), tuple(scaled_landmarks[19]), (0, 0, 255), 2)
    cv2.line(frame, tuple(scaled_landmarks[19]), tuple(scaled_landmarks[20]), (0, 0, 255), 2)

    return frame


def angle_between(v1, v2):
    def unit_vector(vector):
        norm = np.linalg.norm(vector)
        # Protezione contro la divisione per zero
        if norm < 1e-6:
            # Se il vettore è praticamente un punto (norma ~ 0),
            # restituiamo il vettore originale (che è uno zero array)
            return vector
        return vector / norm

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))