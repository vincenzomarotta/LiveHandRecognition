import os
import cv2
import csv
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import mediapipe as mp

# --- MODEL IMPORTS ---
from nn_model.nn_model import GestureClassifierNN
from utils.utility import calculate_finger_angles
from cnn_model.cnn_model import GestureClassifierCNN
from utils.utility import pre_process_landmark, normalize_wrt_keypoint
from utils.metrics import build_cnn_label_map
from heuristic_model.heuristic_model import HeuristicClassifier

# --- PATH CONFIGURATION ---
project_root = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_split', 'test')

NN_WEIGHTS = os.path.join(project_root, "nn_model", "gesture_nn_model.pth")
CNN_WEIGHTS = os.path.join(project_root, "cnn_model", "gesture_cnn_finetuned.pth")
OUTPUT_CSV = os.path.join(project_root, "head_to_head_results.csv")

# --- MODEL INITIALIZATION ---
print("Loading models...")

# 1. NN
nn_classifier = GestureClassifierNN(input_dim=73, num_classes=15)
nn_classifier.load(NN_WEIGHTS)

# 2. CNN
cnn_classifier = GestureClassifierCNN(num_classes=15)
cnn_classifier.model.load_state_dict(torch.load(CNN_WEIGHTS, map_location=cnn_classifier.device))
cnn_classifier.model.eval()

# 3. Heuristic (initialized with a dummy size, updated per frame)
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
    print("\nStarting the 3-way match (CNN vs NN vs Heuristic)...")

    test_path = Path(TEST_DIR)
    results = []
    total_images = 0
    mp_failures = 0

    # Index mapping for ImageFolder (PyTorch alphabetical-order quirk)
    cnn_to_true_label = build_cnn_label_map(TEST_DIR)

    for folder in sorted(test_path.iterdir(), key=lambda x: int(x.name.split('_')[0])):
        if not folder.is_dir(): continue
        true_label = int(folder.name.split('_')[0])
        image_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.png', '.jpeg'}]

        print(f"Processing class {true_label:2d}: {len(image_files)} images...")

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

            # --- 1. CNN PREDICTION ---
            img_tensor = cnn_transform(pil_image).unsqueeze(0).to(cnn_classifier.device)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                cnn_output = cnn_classifier.model(img_tensor)
                cnn_raw_idx = cnn_output.argmax(1).item()
                cnn_pred = cnn_to_true_label[cnn_raw_idx]

            # --- 2. NN AND HEURISTIC PREDICTIONS ---
            results_mp = hands.process(rgb_image_cv)
            nn_pred = -1
            heur_pred = -1

            if results_mp.multi_hand_landmarks:
                hand_landmarks = results_mp.multi_hand_landmarks[0]

                # NN features
                pure_points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                angles = calculate_finger_angles(pure_points)
                landmarks_2d = pre_process_landmark(hand_landmarks, cv_image)
                norm_landmarks = normalize_wrt_keypoint(landmarks_2d, landmarks_2d[0], one_d=True)

                nn_pred = nn_classifier.predict(norm_landmarks + angles)

                # Heuristic input (coordinates scaled to the image's real pixels)
                heur_landmarks = []
                for lm in hand_landmarks.landmark:
                    heur_landmarks.append(np.array([
                        min(int(lm.x * fw), fw - 1),
                        min(int(lm.y * fh), fh - 1),
                        lm.z
                    ], dtype=np.float64))

                # Update the heuristic canvas size for this specific image
                heur_classifier.frame_width = fw
                heur_classifier.frame_height = fh
                heur_pred = heur_classifier.process(heur_landmarks)

            else:
                mp_failures += 1

            # --- 3. ROW SAVE ---
            results.append({
                'img_name': img_path.name,
                'true_label': true_label,
                'cnn_correct': 1 if cnn_pred == true_label else 0,
                'nn_correct': 1 if nn_pred == true_label else 0,
                'euristico_correct': 1 if heur_pred == true_label else 0,
                'mediapipe_failed': 1 if nn_pred == -1 else 0
            })

    hands.close()

    print(f"\nWriting results to: {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 50)
    print(" 3-MODEL TEST SUMMARY ")
    print("=" * 50)
    print(f"Total images          : {total_images}")
    print(f"MediaPipe failures    : {mp_failures}")

    if total_images > 0:
        cnn_acc = sum([r['cnn_correct'] for r in results]) / total_images * 100
        nn_acc = sum([r['nn_correct'] for r in results]) / total_images * 100
        heur_acc = sum([r['euristico_correct'] for r in results]) / total_images * 100

        print(f"Global CNN accuracy   : {cnn_acc:.2f}%")
        print(f"Global NN accuracy    : {nn_acc:.2f}%")
        print(f"Global Heuristic acc. : {heur_acc:.2f}%")


if __name__ == '__main__':
    run_head_to_head_3way()
