import os
import sys
import csv
import time
import multiprocessing as python_mp
from pathlib import Path
from utility import *

# Silence Python-side logs before any MediaPipe/TF import
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import splitfolders

# --- PATH CONFIGURATION ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(project_root, 'dataset', 'hagrid_dataset_512')
OUTPUT_FOLDER = os.path.join(project_root, 'dataset', 'hagrid_dataset_split')
CSV_DIR = os.path.join(project_root, "dataset")

# --- TUNING PARAMETERS ---
NUM_WORKERS = min(8, python_mp.cpu_count())
CHUNK_SIZE = 200
CSV_FLUSH_EVERY = 2_000


def _suppress_stderr():
    """Redirect stderr to /dev/null at the OS level.
    Also blocks logs written directly from C++ by TFLite/MediaPipe."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)


def _normalize_landmarks(landmark_list: list) -> list:
    """Normalize the 3D landmarks (X, Y, Z) with respect to the wrist."""
    base_x, base_y, base_z = landmark_list[0][0], landmark_list[0][1], landmark_list[0][2]

    shifted = []
    for p in landmark_list:
        shifted.extend([
            p[0] - base_x,
            p[1] - base_y,
            p[2] - base_z
        ])

    max_val = max(map(abs, shifted))
    if max_val == 0:
        return shifted

    return [v / max_val for v in shifted]


def _process_chunk(args: tuple) -> list:
    """Process a chunk of images.
    Returns: label + 63 coordinates + 10 angles = 74 columns total."""
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    _suppress_stderr()

    import mediapipe as mp
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    img_paths, label = args

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=0
    )

    rows = []
    try:
        for img_path in img_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            rgb_img.flags.writeable = False
            results = hands.process(rgb_img)
            rgb_img.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Raw points for the angles (keeps the correct 3D proportions)
                    pure_points = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
                    angles = calculate_finger_angles(pure_points)

                    # Pixel-scaled points for positional normalization
                    raw_points = [
                        [
                            min(int(lm.x * w), w - 1),
                            min(int(lm.y * h), h - 1),
                            lm.z
                        ]
                        for lm in hand_lms.landmark
                    ]
                    normalized = _normalize_landmarks(raw_points)

                    # Combine everything: label + 63 spatial features + 10 angle features
                    final_features = normalized + angles
                    rows.append([label, *final_features])
    finally:
        hands.close()

    return rows


def _chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DatasetBuilder:

    def split_images(self) -> bool:
        if not os.path.exists(INPUT_FOLDER):
            print(f"[ERROR] Directory '{INPUT_FOLDER}' not found.")
            return False

        print(f"\n[INFO] Splitting dataset from '{INPUT_FOLDER}'...")
        t0 = time.time()
        splitfolders.ratio(INPUT_FOLDER, output=OUTPUT_FOLDER,
                           seed=1337, ratio=(.8, .1, .1), move=False)
        print(f"[OK] Split complete in {time.time() - t0:.2f}s -> '{OUTPUT_FOLDER}'")
        return True

    def generate_csv_from_split(self, split_name: str) -> None:
        split_path = Path(OUTPUT_FOLDER) / split_name
        csv_file_path = Path(CSV_DIR) / f"{split_name}_keypoint.csv"

        if not split_path.exists():
            print(f"[WARN] Folder '{split_path}' not found. Skipping.")
            return

        print(f"\n[INFO] Generating CSV for '{split_name}' -> {csv_file_path}")
        t_split = time.time()
        Path(CSV_DIR).mkdir(parents=True, exist_ok=True)

        total_images = 0
        total_detected = 0

        with open(csv_file_path, 'w', newline="", buffering=1 << 20) as f:
            writer = csv.writer(f)
            buf = []

            for folder in sorted(split_path.iterdir(), key=lambda x: int(x.name.split('_')[0])):
                if not folder.is_dir():
                    continue

                try:
                    label = int(folder.name.split('_')[0])
                except ValueError:
                    print(f"  [WARN] Invalid folder name: {folder.name}. Skipping.")
                    continue

                img_paths = [
                    str(p) for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                ]
                num_images = len(img_paths)
                print(f"  -> Class {label} ({folder.name}): {num_images} images")

                if num_images == 0:
                    continue

                chunks = list(_chunked(img_paths, CHUNK_SIZE))
                args = [(chunk, label) for chunk in chunks]

                detected_in_class = 0
                t_class = time.time()

                with python_mp.Pool(processes=NUM_WORKERS) as pool:
                    for rows in pool.imap_unordered(_process_chunk, args, chunksize=1):
                        buf.extend(rows)
                        detected_in_class += len(rows)

                        if len(buf) >= CSV_FLUSH_EVERY:
                            writer.writerows(buf)
                            buf.clear()

                if buf:
                    writer.writerows(buf)
                    buf.clear()

                rate = detected_in_class / num_images * 100 if num_images else 0
                print(
                    f"     [OK] {time.time() - t_class:.2f}s | Detected: {detected_in_class}/{num_images} ({rate:.1f}%)")

                total_images += num_images
                total_detected += detected_in_class

        elapsed = time.time() - t_split
        rate = total_detected / total_images * 100 if total_images else 0
        print(
            f"[SUMMARY] '{split_name}' -> {elapsed:.2f}s | {total_detected}/{total_images} hands saved ({rate:.1f}%)")


if __name__ == "__main__":
    python_mp.set_start_method("spawn", force=True)

    t0 = time.time()
    print("=" * 60)
    print(" DATASET BUILDER (COORDS + ANGLES)  ")
    print("=" * 60)
    print(f" Workers  : {NUM_WORKERS}")
    print(f" Chunk    : {CHUNK_SIZE} img/task")
    print("=" * 60)

    builder = DatasetBuilder()

    if builder.split_images():
        for split in ['train', 'val', 'test']:
            builder.generate_csv_from_split(split)

        elapsed = time.time() - t0
        print("\n" + "=" * 60)
        print(" COMPLETE ")
        print(f" Total time : {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        print("=" * 60)
    else:
        print("\n[ERROR] Process aborted: input folder missing.")
