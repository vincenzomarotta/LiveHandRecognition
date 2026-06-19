import mediapipe as mp
import time
import os

import torch
from PIL import Image
from torchvision import transforms

from heuristic_model import heuristic_model
from utils.utility import *
import nn_model.nn_model as nn_model
from cnn_model.cnn_model import GestureClassifierCNN

GESTURE = [
    'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
    'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP',
    'THREE', 'TWO', 'NO GESTURE'
]
HEURISTIC = 0
NN        = 1
CNN       = 2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR     = os.path.join(PROJECT_ROOT, 'dataset', 'hagrid_dataset_split', 'test')

# CNN transform: identica a quella di valutazione (cnn_model.py / test_head_to_head.py)
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def build_cnn_label_map(test_dir):
    """ImageFolder ordina le cartelle alfabeticamente: l'indice grezzo della CNN
    non coincide con la classe reale. Ricostruisce indice_CNN -> label_reale
    leggendo i nomi cartella '{label}_{gesture}'. Fallback identita' se assente."""
    if not os.path.isdir(test_dir):
        print(f"[WARN] {test_dir} non trovato: uso mappa identita' per la CNN.")
        return {i: i for i in range(len(GESTURE))}
    folders = sorted([d for d in os.listdir(test_dir)
                      if os.path.isdir(os.path.join(test_dir, d))])
    return {cnn_idx: int(name.split('_')[0]) for cnn_idx, name in enumerate(folders)}


def open_camera(index=0):
    """Prova indici camera 0 e 1, ritorna il primo funzionante."""
    for idx in [index, 1 - index]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[OK] Camera aperta su indice {idx}")
            return cap
    raise RuntimeError("Nessuna camera trovata agli indici 0 e 1.")


if __name__ == '__main__':
    model_type   = HEURISTIC
    frame_width  = 600
    frame_height = 600

    # Carica SOLO i pesi del modello attivo (gli altri non servono).
    heuristic_classifier = None
    nn_classifier        = None
    cnn_classifier       = None
    cnn_to_true_label    = None

    if model_type == HEURISTIC:
        heuristic_classifier = heuristic_model.HeuristicClassifier((frame_width, frame_height))
    elif model_type == NN:
        nn_classifier = nn_model.GestureClassifierNN()
        nn_classifier.load("./nn_model/gesture_nn_model.pth")
    elif model_type == CNN:
        cnn_classifier = GestureClassifierCNN(num_classes=len(GESTURE))
        cnn_classifier.model.load_state_dict(
            torch.load(os.path.join(PROJECT_ROOT, "cnn_model", "gesture_cnn_finetuned.pth"),
                       map_location=cnn_classifier.device))
        cnn_classifier.model.eval()
        cnn_to_true_label = build_cnn_label_map(TEST_DIR)

    # Etichetta modello (nome, colore BGR) — colori coerenti con la presentazione
    MODEL_LABEL = {
        HEURISTIC: ("HEURISTIC", (111, 149, 242)),  # terra  #F2956F
        NN:        ("NN",        (138, 197,  70)),  # verde  #46C58A
        CNN:       ("CNN",       (240, 155,  76)),  # blu    #4C9BF0
    }
    model_name, model_color = MODEL_LABEL[model_type]

    mp_hands         = mp.solutions.hands
    mp_drawing       = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    last_gesture = 0
    mode         = 0

    # ── FPS tracking ──────────────────────────────────────────────────────────
    fps_counter  = 0
    fps_display  = 0.0
    fps_timer    = time.time()

    cap = open_camera(index=1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    # Riduce il buffer interno: meno latenza, frame più freschi
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    with mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            key = cv2.waitKey(1)   # ← 1ms invece di 10ms: più reattivo
            if key == ord('q'):
                break
            number, mode = select_mode(key, mode)

            ret, frame = cap.read()
            if not ret:
                print("Frame vuoto, riprovo...")
                continue

            frame = cv2.flip(frame, 1)

            # ── Riduzione rumore: filtro Gaussiano passa-basso ────────────────
            # Agisce sull'immagine prima di MediaPipe (euristico/NN) e CNN.
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # ── MediaPipe: processa in RGB ────────────────────────────────────
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame.flags.writeable = True

            gesture_code = 14  # default: NO GESTURE

            # CNN: predice sull'intero frame, non serve MediaPipe
            if model_type == CNN:
                pil = Image.fromarray(rgb)
                tensor = cnn_transform(pil).unsqueeze(0).to(cnn_classifier.device)
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    raw_idx = cnn_classifier.model(tensor).argmax(1).item()
                gesture_code = cnn_to_true_label[raw_idx]

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

                    landmarks = pre_process_landmark(hand_landmarks, frame)

                    if model_type == HEURISTIC:
                        gesture_code = heuristic_classifier.process(landmarks)

                    elif model_type == NN:
                        pure_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        angles      = calculate_finger_angles(pure_points)
                        norm_lm     = normalize_wrt_keypoint(landmarks, landmarks[0], one_d=True)
                        features    = norm_lm + angles
                        gesture_code = nn_classifier.predict(features)

                    brec = bounding_rec(landmarks)
                    frame = draw_info_text(frame, brec, handedness, GESTURE[gesture_code])

                    if last_gesture != gesture_code:
                        print(f"Gesto: {GESTURE[gesture_code]}")
                        last_gesture = gesture_code

            frame = draw_info(frame, mode, number)

            # ── FPS overlay ───────────────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer   = time.time()

            cv2.putText(frame, f"FPS: {fps_display:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # Nome del modello attivo, a sinistra, nel colore della presentazione
            cv2.putText(frame, model_name,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, model_color, 2)

            # ── Mostra il frame SEMPRE, anche senza mani rilevate ─────────────
            cv2.imshow('Hand Gesture Recognition', frame)

    cap.release()
    cv2.destroyAllWindows()