# import os
# import cv2
# import csv
# import time
# import torch
# from pathlib import Path
# from torchvision import transforms
# from PIL import Image
#
# # Import dei tuoi modelli e utility
# from nn_model.nn_model import GestureClassifierNN
# from utils.utility import calculate_finger_angles
# from cnn_model.cnn_model import GestureClassifierCNN  # Il file dove hai messo la CNN MobileNet
# from utils.utility import pre_process_landmark, normalize_wrt_keypoint
#
# import mediapipe as mp
#
# # --- CONFIGURAZIONE PATH ---
# project_root = os.path.dirname(os.path.abspath(__file__))
# TEST_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_split', 'test')
#
# NN_WEIGHTS = os.path.join(project_root, "nn_model", "gesture_nn_model.pth")
# CNN_WEIGHTS = os.path.join(project_root, "cnn_model", "gesture_cnn_finetuned.pth")
# OUTPUT_CSV = os.path.join(project_root, "head_to_head_results.csv")
#
# GESTURE_NAMES = [
#     'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
#     'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP',
#     'THREE', 'TWO', 'NO GESTURE'
# ]
#
# # --- INIZIALIZZAZIONE MODELLI ---
# print("Caricamento modelli in corso...")
#
# # 1. Modello MediaPipe + NN (Keypoints)
# # Assicuriamoci che input_dim sia 73 (63 coordinate + 10 angoli)
# nn_classifier = GestureClassifierNN(input_dim=73, num_classes=15)
# try:
#     nn_classifier.load(NN_WEIGHTS)
#     print("[OK] Pesi NN caricati.")
# except Exception as e:
#     print(f"[ERRORE] Caricamento pesi NN: {e}")
#     exit()
#
# # 2. Modello CNN (MobileNetV3)
# cnn_classifier = GestureClassifierCNN(num_classes=15)
# try:
#     cnn_classifier.model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=cnn_classifier.device))
#     cnn_classifier.model.eval()
#     print("[OK] Pesi CNN caricati.")
# except Exception as e:
#     print(f"[ERRORE] Caricamento pesi CNN: {e}")
#     exit()
#
# # 3. Inizializzazione MediaPipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     model_complexity=0
# )
#
# # 4. Transform per la CNN (IDENTICA a quella del training, ma SENZA ToPILImage)
# cnn_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),  # Si aspetta un'immagine PIL e scala i valori tra 0.0 e 1.0
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
#
# def run_head_to_head_test():
#     print("\nInizio Sfida Diretta (Head-to-Head)...")
#
#     test_path = Path(TEST_DIR)
#     results = []
#
#     # Contatori statistici
#     total_images = 0
#     mp_failures = 0
#
#     # ---------------------------------------------------------
#     # IL FIX: CREIAMO IL DIZIONARIO DI TRADUZIONE PER LA CNN
#     # ---------------------------------------------------------
#     # ImageFolder legge le cartelle in ordine strettamente alfabetico
#     alphabetical_folders = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
#
#     cnn_to_true_label = {}
#     for cnn_idx, folder_name in enumerate(alphabetical_folders):
#         true_label = int(folder_name.split('_')[0])
#         cnn_to_true_label[cnn_idx] = true_label
#
#     print("\n--- Mappatura Indici CNN -> Classi Reali ---")
#     for k, v in cnn_to_true_label.items():
#         print(f"La CNN chiama '{k:2d}' la cartella -> {v:2d} ({alphabetical_folders[k]})")
#     print("--------------------------------------------\n")
#
#     # Navighiamo le cartelle del Test Set per la valutazione
#     for folder in sorted(test_path.iterdir(), key=lambda x: int(x.name.split('_')[0])):
#         if not folder.is_dir(): continue
#
#         true_label = int(folder.name.split('_')[0])
#         image_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.png', '.jpeg'}]
#
#         print(f"Elaborazione classe {true_label:2d} ({folder.name}): {len(image_files)} immagini...")
#
#         for img_path in image_files:
#             try:
#                 pil_image = Image.open(str(img_path)).convert('RGB')
#             except Exception as e:
#                 continue
#
#             cv_image = cv2.imread(str(img_path))
#             if cv_image is None:
#                 continue
#
#             total_images += 1
#             rgb_image_cv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
#
#             # --- 1. PREDIZIONE CNN ---
#             img_tensor = cnn_transform(pil_image).unsqueeze(0).to(cnn_classifier.device)
#             with torch.no_grad():
#                 with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
#                     cnn_output = cnn_classifier.model(img_tensor)
#
#                 # Prendiamo l'indice grezzo e lo TRADUCIAMO
#                 cnn_raw_idx = cnn_output.argmax(1).item()
#                 cnn_pred = cnn_to_true_label[cnn_raw_idx]
#
#             # --- 2. PREDIZIONE MEDIA-PIPE + NN ---
#             results_mp = hands.process(rgb_image_cv)
#             nn_pred = -1
#
#             if results_mp.multi_hand_landmarks:
#                 hand_landmarks = results_mp.multi_hand_landmarks[0]
#
#                 pure_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
#                 angles = calculate_finger_angles(pure_points)
#
#                 landmarks = pre_process_landmark(hand_landmarks, cv_image)
#                 norm_landmarks = normalize_wrt_keypoint(landmarks, landmarks[0], one_d=True)
#
#                 final_features = norm_landmarks + angles
#                 nn_pred = nn_classifier.predict(final_features)
#
#
#
#             else:
#                 mp_failures += 1
#
#
#
#             # --- 3. SALVATAGGIO RIGA ---
#             is_cnn_correct = 1 if cnn_pred == true_label else 0
#             is_nn_correct = 1 if nn_pred == true_label else 0
#
#             results.append({
#                 'img_name': img_path.name,
#                 'true_label': true_label,
#                 'cnn_pred': cnn_pred,
#                 'cnn_correct': is_cnn_correct,
#                 'nn_pred': nn_pred,
#                 'nn_correct': is_nn_correct,
#                 'mediapipe_failed': 1 if nn_pred == -1 else 0
#             })
#
#     hands.close()
#
#     # Scrittura su CSV
#     print(f"\nScrittura risultati in corso su: {OUTPUT_CSV}...")
#     with open(OUTPUT_CSV, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys())
#         writer.writeheader()
#         writer.writerows(results)
#
#     # --- SOMMARIO A SCHERMO ---
#     print("\n" + "=" * 50)
#     print(" SOMMARIO TEST HEAD-TO-HEAD ")
#     print("=" * 50)
#     print(f"Immagini totali analizzate : {total_images}")
#     print(f"Fallimenti MediaPipe       : {mp_failures} ({mp_failures / total_images * 100:.1f}%)")
#
#     if total_images > 0:
#         cnn_accuracy = sum([r['cnn_correct'] for r in results]) / total_images * 100
#         print(f"Precisione Globale CNN     : {cnn_accuracy:.2f}%")
#
#         valid_nn_preds = [r['nn_correct'] for r in results if r['mediapipe_failed'] == 0]
#         if valid_nn_preds:
#             nn_accuracy_on_hands = sum(valid_nn_preds) / len(valid_nn_preds) * 100
#             print(f"Precisione NN (su Mani)    : {nn_accuracy_on_hands:.2f}%")
#
#         nn_global_accuracy = sum([r['nn_correct'] for r in results]) / total_images * 100
#         print(f"Precisione Globale NN      : {nn_global_accuracy:.2f}%")
#
#     print("\n[OK] Test Completato. Puoi ora passare il file a McNemar.")
#     print("=" * 50)
#
# if __name__ == '__main__':
#     run_head_to_head_test()

import os
import cv2
import csv
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import mediapipe as mp

# --- IMPORT DEI TUOI MODELLI ---
from nn_model.nn_model import GestureClassifierNN
from utils.utility import calculate_finger_angles
from cnn_model.cnn_model import GestureClassifierCNN
from utils.utility import pre_process_landmark, normalize_wrt_keypoint

# IMPORTA IL TUO MODELLO EURISTICO (Assicurati che il nome del file sia corretto)
from heuristic_model.heuristic_model import HeuristicClassifier

# --- CONFIGURAZIONE PATH ---
project_root = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_split', 'test')

NN_WEIGHTS = os.path.join(project_root, "nn_model", "gesture_nn_model.pth")
CNN_WEIGHTS = os.path.join(project_root, "cnn_model", "gesture_cnn_finetuned.pth")
OUTPUT_CSV = os.path.join(project_root, "head_to_head_results.csv")

# --- INIZIALIZZAZIONE MODELLI ---
print("Caricamento modelli in corso...")

# 1. NN
nn_classifier = GestureClassifierNN(input_dim=73, num_classes=15)
nn_classifier.load(NN_WEIGHTS)

# 2. CNN
cnn_classifier = GestureClassifierCNN(num_classes=15)
cnn_classifier.model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=cnn_classifier.device))
cnn_classifier.model.eval()

# 3. Euristico (Inizializzato con una dimensione dummy, verrà aggiornato frame per frame)
heur_classifier = HeuristicClassifier((1920, 1080))

# 4. MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1,
    min_detection_confidence=0.5, model_complexity=0
)

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def run_head_to_head_3way():
    print("\nInizio Sfida a 3 (CNN vs NN vs Euristico)...")

    test_path = Path(TEST_DIR)
    results = []
    total_images = 0
    mp_failures = 0

    # Mappatura indici per ImageFolder (Bug Alfabetico PyTorch)
    alphabetical_folders = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
    cnn_to_true_label = {idx: int(name.split('_')[0]) for idx, name in enumerate(alphabetical_folders)}

    for folder in sorted(test_path.iterdir(), key=lambda x: int(x.name.split('_')[0])):
        if not folder.is_dir(): continue
        true_label = int(folder.name.split('_')[0])
        image_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.png', '.jpeg'}]

        print(f"Elaborazione classe {true_label:2d}: {len(image_files)} immagini...")

        for img_path in image_files:
            try:
                pil_image = Image.open(str(img_path)).convert('RGB')
            except Exception:
                continue

            cv_image = cv2.imread(str(img_path))
            if cv_image is None: continue

            total_images += 1
            fh, fw, _ = cv_image.shape
            rgb_image_cv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # --- 1. PREDIZIONE CNN ---
            img_tensor = cnn_transform(pil_image).unsqueeze(0).to(cnn_classifier.device)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                cnn_output = cnn_classifier.model(img_tensor)
                cnn_raw_idx = cnn_output.argmax(1).item()
                cnn_pred = cnn_to_true_label[cnn_raw_idx]

            # --- 2. PREDIZIONE NN E EURISTICO ---
            results_mp = hands.process(rgb_image_cv)
            nn_pred = -1
            heur_pred = -1

            if results_mp.multi_hand_landmarks:
                hand_landmarks = results_mp.multi_hand_landmarks[0]

                # Setup per la NN (Features)
                pure_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                angles = calculate_finger_angles(pure_points)
                landmarks_2d = pre_process_landmark(hand_landmarks, cv_image)
                norm_landmarks = normalize_wrt_keypoint(landmarks_2d, landmarks_2d[0], one_d=True)

                nn_pred = nn_classifier.predict(norm_landmarks + angles)

                # Setup per l'Euristico (Coordinate scalate ai pixel reali dell'immagine)
                heur_landmarks = []
                for lm in hand_landmarks.landmark:
                    heur_landmarks.append(np.array([
                        min(int(lm.x * fw), fw - 1),
                        min(int(lm.y * fh), fh - 1),
                        lm.z
                    ], dtype=np.float64))

                # Aggiorniamo le dimensioni del canvas dell'euristico per questa specifica immagine
                heur_classifier.frame_width = fw
                heur_classifier.frame_height = fh
                heur_pred = heur_classifier.process(heur_landmarks)

            else:
                mp_failures += 1

            # --- 3. SALVATAGGIO RIGA ---
            results.append({
                'img_name': img_path.name,
                'true_label': true_label,
                'cnn_correct': 1 if cnn_pred == true_label else 0,
                'nn_correct': 1 if nn_pred == true_label else 0,
                'euristico_correct': 1 if heur_pred == true_label else 0,
                'mediapipe_failed': 1 if nn_pred == -1 else 0
            })

    hands.close()

    print(f"\nScrittura risultati su: {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 50)
    print(" SOMMARIO TEST A 3 MODELLI ")
    print("=" * 50)
    print(f"Immagini totali            : {total_images}")
    print(f"Fallimenti MediaPipe       : {mp_failures}")

    if total_images > 0:
        cnn_acc = sum([r['cnn_correct'] for r in results]) / total_images * 100
        nn_acc = sum([r['nn_correct'] for r in results]) / total_images * 100
        heur_acc = sum([r['euristico_correct'] for r in results]) / total_images * 100

        print(f"Precisione Globale CNN     : {cnn_acc:.2f}%")
        print(f"Precisione Globale NN      : {nn_acc:.2f}%")
        print(f"Precisione Globale Eurist. : {heur_acc:.2f}%")


if __name__ == '__main__':
    run_head_to_head_3way()