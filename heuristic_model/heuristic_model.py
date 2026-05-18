# import os
# from utils.utility import *
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import math
# import numpy as np
# import concurrent.futures
#
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TRAIN_FILE = os.path.join(project_root, "dataset", "train_keypoint.csv")
# VAL_FILE = os.path.join(project_root, "dataset", "val_keypoint.csv")
# TEST_FILE = os.path.join(project_root, "dataset", "test_keypoint.csv")
#
# GESTURE_NAMES = [
#     'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
#     'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP', 'THREE', 'TWO', 'NO GESTURE'
# ]
#
# class HeuristicClassifier:
#     NOSTATE = 0
#     STRAIGHT = 1
#     BENT = 2
#     CROSSED = 3
#
#     CALL = 0
#     DISLIKE = 1
#     CLOSED_FIST = 2
#     FOUR = 3
#     LIKE = 4
#     MUTE = 5
#     OK = 6
#     POINTING_UP = 7
#     OPEN_PALM = 8
#     VICTORY = 9
#     ROCK = 10
#     STOP = 11
#     THREE = 12
#     TWO = 13
#     NO_GESTURE = 14
#
#
#     def __init__(self, frame_size):
#         self.frame_width = frame_size[0]
#         self.frame_height = frame_size[1]
#
#         self._pts2 = None
#         self._ref_shape = None
#         self._homo_valid = False
#         self._init_homography_reference()
#
#
#     def process(self, landmarks):
#
#         hand_img = np.ones((self.frame_width, self.frame_height)) * 255
#         hand_img = draw_hand(hand_img, landmarks)
#
#         n_fingers = 5
#         state = np.ones(n_fingers) * -1
#
#         # --- FIX: Calcoliamo il centro mantenendo tutte e 3 le coordinate ---
#         center_kp = (landmarks[5] + landmarks[9] + landmarks[17]) / 3.0
#         norm_landmarks = normalize_wrt_keypoint(landmarks, center_kp)
#
#         # Per calcolare le distanze 2D successive, impostiamo l'origine a (0,0)
#         origin_2d = np.zeros(2)
#
#         lm_index = 4
#         for i in range(n_fingers):
#             lm = norm_landmarks[lm_index]
#             # Usiamo origin_2d per calcolare la distanza planare
#             d = math.sqrt((lm[0] - origin_2d[0]) ** 2 + (lm[1] - origin_2d[1]) ** 2)
#
#             if (d > 0.6):
#                 state[i] = self.STRAIGHT
#             else:
#                 state[i] = self.BENT
#
#             lm_index = lm_index + 4
#         # --- NUOVE EURISTICHE ---
#
#         # Compute the angle (La tua logica originale per il pollice incrociato)
#         h, _ = self.perspective_correction(hand_img, landmarks)
#         if h is not None:
#             to_transform = [landmarks[0], landmarks[1], landmarks[5]]
#             transformed = self.transform_points(to_transform, h)
#             norm_transformed_landmarks = normalize_wrt_keypoint(transformed, transformed[0])
#
#             v1 = norm_transformed_landmarks[1]
#             v2 = norm_transformed_landmarks[2]
#             angle = angle_between(v1, v2)
#
#             if angle < 0.8:
#                 state[0] = self.CROSSED
#
#         # Passiamo stato e landmarks direttamente al classificatore
#         hand_gesture = self.__gesture(state, landmarks)
#
#         return hand_gesture
#
#
#     def _init_homography_reference(self):
#         """Carica HomographyRef.png e calcola i pts2 una volta sola."""
#         current_dir  = os.path.dirname(os.path.abspath(__file__))
#         ref_img_path = os.path.join(current_dir, 'HomographyRef.png')
#         ref_img      = cv2.imread(ref_img_path, 0)
#
#         if ref_img is None:
#             print(f"[WARN] HomographyRef.png non trovato in: {ref_img_path}")
#             return
#
#         circles = cv2.HoughCircles(
#             ref_img, cv2.HOUGH_GRADIENT, 1, 46,
#             param1=50, param2=1, minRadius=4, maxRadius=30
#         )
#         if circles is None:
#             print("[WARN] Nessun cerchio trovato in HomographyRef.png")
#             return
#
#         dict_circles = {}
#         for (x, y, r) in np.uint16(np.round(circles[0, :])):
#             dict_circles[r] = (x, y)
#
#         self._pts2       = np.float32([v for _, v in sorted(dict_circles.items())])
#         self._ref_shape  = ref_img.shape
#         self._homo_valid = True
#         print(f"[OK] Homography reference caricata ({len(self._pts2)} punti)")
#
#
#     def perspective_correction(self, frame, landmarks):
#
#         if not self._homo_valid:
#             return None, frame
#
#         pts1 = np.float32([[landmarks[0][:2]],
#                            [landmarks[5][:2]],
#                            [landmarks[9][:2]],
#                            [landmarks[17][:2]]])
#
#         h = cv2.getPerspectiveTransform(pts1, self._pts2[:4])
#         im_out = cv2.warpPerspective(
#             frame, h, (self._ref_shape[1], self._ref_shape[0])
#         )
#         return h, im_out
#
#
#     def transform_points(self, landmarks, h):
#         points = []
#         for lm in landmarks:
#             points.append([[lm[0], lm[1]]])
#         points = np.float32(points)
#
#         transformed = cv2.perspectiveTransform(points, h).squeeze()
#
#         # --- FIX: Aggiungiamo una colonna di zeri (Z=0) per mantenere il formato 3D ---
#         transformed_3d = np.zeros((transformed.shape[0], 3))
#         transformed_3d[:, :2] = transformed
#
#         return transformed_3d
#
#
#     def __gesture(self, state, landmarks):
#         thumb_state = state[0]
#         index_state = state[1]
#         middle_state = state[2]
#         ring_state = state[3]
#         pinky_state = state[4]
#
#         # Funzione helper per calcolare la distanza euclidea tra due keypoints
#         def dist(lm1, lm2):
#             return math.sqrt(
#                 (landmarks[lm1][0] - landmarks[lm2][0]) ** 2 + (landmarks[lm1][1] - landmarks[lm2][1]) ** 2)
#
#         # Larghezza del palmo come unità di misura base
#         palm_width = dist(5, 17) + 1e-6
#
#         # ==========================================
#         # 1. GESTI DI PRECISIONE (Priorità Alta)
#         # ==========================================
#
#         # OK / MUTE (Pizzico molto stretto tra pollice e indice)
#         pinch_dist = dist(4, 8) / palm_width
#         if pinch_dist < 0.35:  # Soglia corretta (prima era 0.8, troppo grande)
#             if middle_state == self.STRAIGHT and ring_state == self.STRAIGHT:
#                 return self.OK
#             else:
#                 return self.MUTE
#
#         # LIKE / DISLIKE (Indice, medio, anulare piegati. Pollice dritto)
#         if index_state == self.BENT and middle_state == self.BENT and ring_state == self.BENT:
#             if thumb_state == self.STRAIGHT and pinky_state == self.BENT:
#                 if landmarks[4][1] < landmarks[2][1]:  # Punta del pollice sopra la base
#                     return self.LIKE
#                 else:
#                     return self.DISLIKE
#
#         # CALL (Pollice e mignolo dritti, in mezzo tutto piegato)
#         if thumb_state == self.STRAIGHT and pinky_state == self.STRAIGHT and index_state == self.BENT and middle_state == self.BENT:
#             return self.CALL
#
#         # ROCK / CORNA (Indice e mignolo dritti, medio e anulare piegati)
#         if index_state == self.STRAIGHT and pinky_state == self.STRAIGHT and middle_state == self.BENT and ring_state == self.BENT:
#             return self.ROCK
#
#         # ==========================================
#         # 2. COMBINAZIONI STANDARD DITA (Priorità Media)
#         # ==========================================
#
#         # POINTING_UP (Solo indice dritto)
#         if index_state == self.STRAIGHT and middle_state == self.BENT and ring_state == self.BENT and pinky_state == self.BENT:
#             return self.POINTING_UP
#
#         # VICTORY / TWO (Indice e medio dritti)
#         if index_state == self.STRAIGHT and middle_state == self.STRAIGHT and ring_state == self.BENT and pinky_state == self.BENT:
#             fingers_dist = dist(8, 12) / palm_width
#             if fingers_dist > 0.6:  # Dita a V
#                 return self.VICTORY
#             else:  # Dita unite/parallele
#                 return self.TWO
#
#         # THREE (Tre dita centrali dritte)
#         if index_state == self.STRAIGHT and middle_state == self.STRAIGHT and ring_state == self.STRAIGHT and pinky_state == self.BENT:
#             return self.THREE
#
#         # ==========================================
#         # 3. GESTI "PIGLIATUTTO" (Priorità Bassa)
#         # ==========================================
#
#         # FIST (Tutte le 4 dita lunghe piegate)
#         # Arrivati qui sappiamo che NON è Like, Dislike o Call, quindi possiamo essere generici
#         if index_state == self.BENT and middle_state == self.BENT and ring_state == self.BENT and pinky_state == self.BENT:
#             return self.CLOSED_FIST
#
#         # FOUR vs OPEN_PALM vs STOP (Tutte le 4 dita lunghe dritte)
#         if index_state == self.STRAIGHT and middle_state == self.STRAIGHT and ring_state == self.STRAIGHT and pinky_state == self.STRAIGHT:
#             # Distinguiamo tramite il pollice
#             if thumb_state in (self.BENT, self.CROSSED):
#                 return self.FOUR
#             else:
#                 # Se anche il pollice è dritto, è OPEN_PALM o STOP.
#                 # STOP di solito ha le dita molto separate (tensione). OPEN_PALM è più rilassata.
#                 spread = dist(8, 20) / palm_width  # Distanza tra indice e mignolo
#                 if spread > 1.3:
#                     return self.STOP
#                 else:
#                     return self.OPEN_PALM
#
#         return self.NO_GESTURE
#
#     def evaluate(self, X_test, y_test, matrix=True):
#         frame_width = self.frame_height
#         frame_height = self.frame_width
#
#         n_landmarks = X_test.shape[0]
#
#         print(f"\n[INFO] --- Starting Heuristic Evaluation ---")
#         print(f"[INFO] Total samples to evaluate: {n_landmarks}")
#         print(f"[INFO] Pre-processing landmarks per il calcolo parallelo...")
#
#         # --- FASE 1: Prepariamo tutti i landmark in una grande lista ---
#         all_landmarks = []
#         for i in range(n_landmarks):
#             coords_3d = X_test[i][:63].reshape((21, 3))
#             landmarks = []
#             for j in range(21):
#                 lm = coords_3d[j]
#                 lm_x = min(int(lm[0] * frame_width), frame_width - 1)
#                 lm_y = min(int(lm[1] * frame_height), frame_height - 1)
#                 lm_z = lm[2]
#                 landmarks.append(np.array([lm_x, lm_y, lm_z], dtype=np.float64))
#
#             all_landmarks.append(landmarks)
#
#         print(f"[INFO] Avvio elaborazione su tutti i core della CPU...")
#         y_pred = []
#
#         # Sfrutta automaticamente tutti i core disponibili sul tuo PC
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             # map esegue self.process su ogni elemento di all_landmarks in parallelo
#             # ed è garantito che l'ordine dei risultati rispetti l'ordine di y_test!
#             results = executor.map(self.process, all_landmarks)
#
#             # Convertiamo il generatore in una lista
#             y_pred = list(results)
#
#         y_pred = np.array(y_pred)
#
#         print(f"\n[INFO] Landmark processing complete. Generating metrics...")
#
#         # Compute the confusion matrix (rimane invariato)
#         def print_confusion_matrix(y_true, y_pred_labels):
#             print("[INFO] Rendering confusion matrix plot...")
#             labels = sorted(list(set(y_true) | set(y_pred_labels)))
#             cmx_data = confusion_matrix(y_true, y_pred_labels, labels=labels)
#
#             display_labels = [GESTURE_NAMES[l] for l in labels if l < len(GESTURE_NAMES)]
#
#             df_cmx = pd.DataFrame(cmx_data, index=display_labels, columns=display_labels)
#
#             fig, ax = plt.subplots(figsize=(10, 8))
#             sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
#             ax.set_ylim(len(display_labels), 0)
#             plt.title('Confusion Matrix - Heuristic Classifier')
#             plt.show()
#
#         print('\n--- Classification Report ---')
#         print(classification_report(y_test, y_pred, target_names=[GESTURE_NAMES[i] for i in sorted(list(set(y_test)))],
#                                     zero_division=0))
#
#         if matrix:
#             print_confusion_matrix(y_test, y_pred)
#
#         print("[INFO] Evaluation finished successfully.\n")
#
#
# if __name__ == '__main__':
#     # Run the current file in order to evaluate the heuristic classifier
#     RANDOM_SEED = 142
#
#     # x, y, z and 10 angles
#     X_dataset = np.loadtxt(TRAIN_FILE, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 3 + 10) + 1)))
#     y_dataset = np.loadtxt(TRAIN_FILE, delimiter=',', dtype='int32', usecols=0)
#
#     X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
#
#     model = HeuristicClassifier((1280, 720))
#     model.evaluate(X_test, y_test)

import os
import math
import time
import csv
import numpy as np
import concurrent.futures
from utils.utility import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from nn_model.load_and_plot import plot_confusion_matrix, save_metrics_csv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(project_root, "dataset", "train_keypoint.csv")
VAL_FILE   = os.path.join(project_root, "dataset", "val_keypoint.csv")
TEST_FILE  = os.path.join(project_root, "dataset", "test_keypoint.csv")

GESTURE_NAMES = [
    'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
    'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP', 'THREE', 'TWO', 'NO GESTURE'
]


class HeuristicClassifier:
    NOSTATE = 0; STRAIGHT = 1; BENT = 2; CROSSED = 3
    CALL=0; DISLIKE=1; CLOSED_FIST=2; FOUR=3; LIKE=4; MUTE=5; OK=6
    POINTING_UP=7; OPEN_PALM=8; VICTORY=9; ROCK=10; STOP=11
    THREE=12; TWO=13; NO_GESTURE=14

    def __init__(self, frame_size):
        self.frame_width  = frame_size[0]
        self.frame_height = frame_size[1]
        self._pts2       = None
        self._ref_shape  = None
        self._homo_valid = False
        self._init_homography_reference()

    def _init_homography_reference(self):
        current_dir  = os.path.dirname(os.path.abspath(__file__))
        ref_img_path = os.path.join(current_dir, 'HomographyRef.png')
        ref_img      = cv2.imread(ref_img_path, 0)
        if ref_img is None:
            print(f"[WARN] HomographyRef.png non trovato in: {ref_img_path}")
            return
        circles = cv2.HoughCircles(
            ref_img, cv2.HOUGH_GRADIENT, 1, 46,
            param1=50, param2=1, minRadius=4, maxRadius=30
        )
        if circles is None:
            print("[WARN] Nessun cerchio trovato in HomographyRef.png")
            return
        dict_circles = {}
        for (x, y, r) in np.uint16(np.round(circles[0, :])):
            dict_circles[r] = (x, y)
        self._pts2      = np.float32([v for _, v in sorted(dict_circles.items())])
        self._ref_shape = ref_img.shape
        self._homo_valid = True
        print(f"[OK] Homography reference caricata ({len(self._pts2)} punti)")

    def perspective_correction(self, frame, landmarks):
        if not self._homo_valid:
            return None, frame
        pts1 = np.float32([[landmarks[0][:2]], [landmarks[5][:2]],
                            [landmarks[9][:2]], [landmarks[17][:2]]])
        h      = cv2.getPerspectiveTransform(pts1, self._pts2[:4])
        im_out = cv2.warpPerspective(frame, h, (self._ref_shape[1], self._ref_shape[0]))
        return h, im_out

    def transform_points(self, landmarks, h):
        points = np.float32([[[lm[0], lm[1]]] for lm in landmarks])
        transformed    = cv2.perspectiveTransform(points, h).squeeze()
        transformed_3d = np.zeros((transformed.shape[0], 3))
        transformed_3d[:, :2] = transformed
        return transformed_3d

    def process(self, landmarks):
        hand_img   = np.ones((self.frame_width, self.frame_height)) * 255
        hand_img   = draw_hand(hand_img, landmarks)
        state      = np.ones(5) * -1
        center_kp  = (landmarks[5] + landmarks[9] + landmarks[17]) / 3.0
        norm_lm    = normalize_wrt_keypoint(landmarks, center_kp)
        origin_2d  = np.zeros(2)

        for i, lm_idx in enumerate([4, 8, 12, 16, 20]):
            lm     = norm_lm[lm_idx]
            d      = math.hypot(lm[0] - origin_2d[0], lm[1] - origin_2d[1])
            state[i] = self.STRAIGHT if d > 0.6 else self.BENT

        h, _ = self.perspective_correction(hand_img, landmarks)
        if h is not None:
            transformed = self.transform_points([landmarks[0], landmarks[1], landmarks[5]], h)
            norm_tr     = normalize_wrt_keypoint(transformed, transformed[0])
            if angle_between(norm_tr[1], norm_tr[2]) < 0.8:
                state[0] = self.CROSSED

        return self.__gesture(state, landmarks)

    def __gesture(self, state, landmarks):
        thumb, index, middle, ring, pinky = state

        def dist(a, b):
            return math.hypot(landmarks[a][0] - landmarks[b][0],
                              landmarks[a][1] - landmarks[b][1])

        palm_width = dist(5, 17) + 1e-6
        pinch_dist = dist(4, 8) / palm_width

        if pinch_dist < 0.35:
            return self.OK if (middle == self.STRAIGHT and ring == self.STRAIGHT) else self.MUTE
        if index == self.BENT and middle == self.BENT and ring == self.BENT:
            if thumb == self.STRAIGHT and pinky == self.BENT:
                return self.LIKE if landmarks[4][1] < landmarks[2][1] else self.DISLIKE
        if thumb == self.STRAIGHT and pinky == self.STRAIGHT and index == self.BENT and middle == self.BENT:
            return self.CALL
        if index == self.STRAIGHT and pinky == self.STRAIGHT and middle == self.BENT and ring == self.BENT:
            return self.ROCK
        if index == self.STRAIGHT and middle == self.BENT and ring == self.BENT and pinky == self.BENT:
            return self.POINTING_UP
        if index == self.STRAIGHT and middle == self.STRAIGHT and ring == self.BENT and pinky == self.BENT:
            return self.VICTORY if dist(8, 12) / palm_width > 0.6 else self.TWO
        if index == self.STRAIGHT and middle == self.STRAIGHT and ring == self.STRAIGHT and pinky == self.BENT:
            return self.THREE
        if index == self.BENT and middle == self.BENT and ring == self.BENT and pinky == self.BENT:
            return self.CLOSED_FIST
        if index == self.STRAIGHT and middle == self.STRAIGHT and ring == self.STRAIGHT and pinky == self.STRAIGHT:
            if thumb in (self.BENT, self.CROSSED):
                return self.FOUR
            return self.STOP if dist(8, 20) / palm_width > 1.3 else self.OPEN_PALM
        return self.NO_GESTURE

    def _preprocess_row(self, row):
        """Converte una riga del CSV in lista di landmark 3D."""
        fw, fh    = self.frame_height, self.frame_width
        coords_3d = row[:63].reshape((21, 3))
        landmarks = []
        for j in range(21):
            lm = coords_3d[j]
            landmarks.append(np.array([
                min(int(lm[0] * fw), fw - 1),
                min(int(lm[1] * fh), fh - 1),
                lm[2]
            ], dtype=np.float64))
        return landmarks

    def evaluate(self, X_test, y_test, matrix=True, n_jobs=4,
                 output_csv="test_results_heuristic.csv",
                 metrics_csv="metrics_summary_heuristic.csv"):
        """
        Valuta il classificatore euristico producendo:
        - CSV per-sample con is_correct e latency_ms  (per McNemar)
        - CSV metriche aggregate (precision/recall/f1 + latenza media)
        - Confusion matrix salvata su disco
        """
        n = X_test.shape[0]
        print(f"\n[INFO] Valutazione su {n} sample con {n_jobs} thread...")

        # ── Pre-processing landmark ───────────────────────────────────────────
        all_landmarks = [self._preprocess_row(X_test[i]) for i in range(n)]

        # ── Inferenza parallela con latenza per-sample ────────────────────────
        # Misuriamo la latenza su ogni singolo sample perché nell'uso reale
        # il classificatore riceve un frame alla volta dal loop webcam.
        results_raw = [None] * n

        def process_timed(idx):
            lm = all_landmarks[idx]
            t0 = time.perf_counter()
            pred = self.process(lm)
            lat  = (time.perf_counter() - t0) * 1000
            return idx, pred, lat

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for idx, pred, lat in ex.map(process_timed, range(n)):
                results_raw[idx] = (pred, lat)

        y_pred    = np.array([r[0] for r in results_raw])
        latencies = np.array([r[1] for r in results_raw])
        avg_lat   = float(np.mean(latencies))
        print(f"Latenza media per sample: {avg_lat:.4f} ms")

        # ── CSV per-sample (McNemar) ──────────────────────────────────────────
        rows = [
            {
                'sample_index': i,
                'true_label':   int(y_test[i]),
                'pred_label':   int(y_pred[i]),
                'is_correct':   1 if y_pred[i] == y_test[i] else 0,
                'latency_ms':   round(latencies[i], 4)
            }
            for i in range(n)
        ]
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] {n} risultati per-sample salvati in: {output_csv}")

        # ── CSV metriche aggregate ────────────────────────────────────────────
        save_metrics_csv(
            y_test, y_pred,
            class_names=GESTURE_NAMES,
            avg_latency_ms=avg_lat,
            output_csv=metrics_csv
        )

        # ── Report testuale ───────────────────────────────────────────────────
        present = sorted(set(y_test))
        print('\n--- Classification Report ---')
        print(classification_report(
            y_test, y_pred,
            labels=present,
            target_names=[GESTURE_NAMES[i] for i in present],
            zero_division=0
        ))

        # ── Confusion matrix ──────────────────────────────────────────────────
        if matrix:
            cm = confusion_matrix(y_test, y_pred, labels=present)
            plot_confusion_matrix(
                cm,
                class_names=[GESTURE_NAMES[i] for i in present],
                save_path="plots/heuristic_confusion_matrix.png"
            )

        print("[INFO] Valutazione completata.\n")


if __name__ == '__main__':

    X_dataset = np.loadtxt(TRAIN_FILE, delimiter=',', dtype='float32',
                           usecols=list(range(1, (21 * 3 + 10) + 1)))
    y_dataset = np.loadtxt(TRAIN_FILE, delimiter=',', dtype='int32', usecols=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, y_dataset, train_size=0.75, random_state=142
    )

    model = HeuristicClassifier((1280, 720))
    model.evaluate(
        X_test, y_test,
        n_jobs=4,
        output_csv="test_results_heuristic.csv",
        metrics_csv="metrics_summary_heuristic.csv"
    )