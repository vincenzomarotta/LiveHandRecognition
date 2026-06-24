import mediapipe as mp
import time
import os
from collections import deque, Counter

import torch
from PIL import Image
from torchvision import transforms

from heuristic_model import heuristic_model
from utils.utility import *
from utils.metrics import GESTURE_NAMES as GESTURE, build_cnn_label_map
import nn_model.nn_model as nn_model
from cnn_model.cnn_model import GestureClassifierCNN

HEURISTIC = 0
NN = 1
CNN = 2
NO_GESTURE = len(GESTURE) - 1  # index 14

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'hagrid_dataset_split', 'test')

WEIGHTS = {
    NN:  os.path.join(PROJECT_ROOT, "nn_model", "gesture_nn_model.pth"),
    CNN: os.path.join(PROJECT_ROOT, "cnn_model", "gesture_cnn_finetuned.pth"),
}

# Number of frames in the majority-vote smoothing window
SMOOTH_WINDOW = 7

# CNN transform: same as the evaluation pipeline (cnn_model.py / test_head_to_head.py)
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model label (name, BGR color) matching the presentation palette
MODEL_LABEL = {
    HEURISTIC: ("HEURISTIC", (111, 149, 242)),
    NN:        ("NN",        (138, 197,  70)),
    CNN:       ("CNN",       (240, 155,  76)),
}


class GestureSmoother:
    """Majority vote over the last N frames: removes single-frame flicker."""

    def __init__(self, window):
        self.buf = deque(maxlen=window)

    def update(self, gesture_code):
        self.buf.append(gesture_code)
        return Counter(self.buf).most_common(1)[0][0]

    def reset(self):
        self.buf.clear()


def open_camera(index=0):
    """Try camera indices 0 and 1, return the first one that opens."""
    for idx in [index, 1 - index]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[OK] Camera opened on index {idx}")
            return cap
    raise RuntimeError("No camera found on indices 0 and 1.")


def load_classifier(model_type, frame_size):
    """Build a classifier on demand. Returns (classifier, cnn_label_map).
    Raises FileNotFoundError if the trained weights are missing."""
    if model_type == HEURISTIC:
        return heuristic_model.HeuristicClassifier(frame_size), None

    weights = WEIGHTS[model_type]
    if not os.path.isfile(weights):
        raise FileNotFoundError(weights)

    if model_type == NN:
        clf = nn_model.GestureClassifierNN()
        clf.load(weights)
        return clf, None

    # CNN
    clf = GestureClassifierCNN(num_classes=len(GESTURE))
    clf.model.load_state_dict(torch.load(weights, map_location=clf.device))
    clf.model.eval()
    return clf, build_cnn_label_map(TEST_DIR)


def predict(model_type, classifier, cnn_label_map, rgb, hand_landmarks, frame):
    """Run one prediction for the active model. The CNN ignores the landmarks
    and reads the whole frame; the other two work on the hand geometry."""
    if model_type == CNN:
        tensor = cnn_transform(Image.fromarray(rgb)).unsqueeze(0).to(classifier.device)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            raw_idx = classifier.model(tensor).argmax(1).item()
        return cnn_label_map[raw_idx]

    landmarks = pre_process_landmark(hand_landmarks, frame)
    if model_type == HEURISTIC:
        return classifier.process(landmarks)

    # NN
    pure_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    angles = calculate_finger_angles(pure_points)
    features = normalize_wrt_keypoint(landmarks, landmarks[0], one_d=True) + angles
    return classifier.predict(features)


if __name__ == '__main__':
    frame_width = 600
    frame_height = 600
    frame_size = (frame_width, frame_height)

    # Lazy cache: each model is loaded the first time it is selected.
    cache = {}

    def get_classifier(model_type):
        if model_type not in cache:
            cache[model_type] = load_classifier(model_type, frame_size)
        return cache[model_type]

    model_type = HEURISTIC
    classifier, cnn_label_map = get_classifier(model_type)
    model_name, model_color = MODEL_LABEL[model_type]

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    smoother = GestureSmoother(SMOOTH_WINDOW)
    last_gesture = NO_GESTURE

    # FPS tracking
    fps_counter = 0
    fps_display = 0.0
    fps_timer = time.time()

    cap = open_camera(index=1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    # Shrink the internal buffer for lower latency and fresher frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Key -> model for live switching
    switch_keys = {ord('1'): HEURISTIC, ord('2'): NN, ord('3'): CNN}

    with mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            key = cv2.waitKey(1)  # 1ms wait keeps the loop responsive
            if key == ord('q'):
                break

            # Live model switch (1=heuristic, 2=NN, 3=CNN)
            if key in switch_keys and switch_keys[key] != model_type:
                requested = switch_keys[key]
                try:
                    classifier, cnn_label_map = get_classifier(requested)
                    model_type = requested
                    model_name, model_color = MODEL_LABEL[model_type]
                    smoother.reset()
                    print(f"[OK] Switched to {model_name}")
                except FileNotFoundError as e:
                    print(f"[WARN] Cannot switch to {MODEL_LABEL[requested][0]}: weights not found ({e})")

            ret, frame = cap.read()
            if not ret:
                print("Empty frame, retrying...")
                continue

            frame = cv2.flip(frame, 1)

            # Noise reduction: Gaussian low-pass filter applied before MediaPipe and the CNN
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # MediaPipe expects RGB
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame.flags.writeable = True

            gesture_code = NO_GESTURE

            # CNN predicts on the whole frame, regardless of detection
            if model_type == CNN:
                gesture_code = predict(model_type, classifier, cnn_label_map, rgb, None, frame)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness):

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    if model_type != CNN:
                        gesture_code = predict(model_type, classifier, cnn_label_map, rgb, hand_landmarks, frame)

                    brec = bounding_rec(pre_process_landmark(hand_landmarks, frame))
                    frame = draw_info_text(frame, brec, handedness, GESTURE[gesture_code])

            # Temporal smoothing of the frame-level prediction
            stable_gesture = smoother.update(gesture_code)
            if stable_gesture != last_gesture:
                print(f"Gesture: {GESTURE[stable_gesture]}")
                last_gesture = stable_gesture

            # FPS overlay
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            cv2.putText(frame, f"FPS: {fps_display:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # Active model name on the left, in the presentation color
            cv2.putText(frame, model_name,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, model_color, 2)

            # Controls hint
            cv2.putText(frame, "1 HEUR  2 NN  3 CNN  q QUIT",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1, cv2.LINE_AA)

            # Big smoothed gesture label, centered at the bottom
            label = GESTURE[stable_gesture]
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.putText(frame, label,
                        ((frame.shape[1] - tw) // 2, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, model_color, 3, cv2.LINE_AA)

            # Always show the frame, even when no hand is detected
            cv2.imshow('Hand Gesture Recognition', frame)

    cap.release()
    cv2.destroyAllWindows()
