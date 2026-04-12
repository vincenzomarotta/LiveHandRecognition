import os
import sys
import csv
import time
import multiprocessing as python_mp
from pathlib import Path

# Silenzia i log Python-side prima di qualsiasi import MP/TF
os.environ["GLOG_minloglevel"]     = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import splitfolders

# --- PATH CONFIGURATION ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(project_root, 'dataset', 'hagrid_dataset_512')
OUTPUT_FOLDER = os.path.join(project_root, 'dataset', 'hagrid_dataset_splitted')
CSV_DIR = os.path.join(project_root, "dataset")

# --- TUNING PARAMETERS ---
NUM_WORKERS = min(8, python_mp.cpu_count())
CHUNK_SIZE = 200
CSV_FLUSH_EVERY = 2_000

# ---------------------------------------------------------------------------
# Funzioni standalone (devono essere pickle-able per multiprocessing)
# ---------------------------------------------------------------------------

def _suppress_stderr():
    """
    Redirige stderr a /dev/null a livello di OS.
    Blocca anche i log scritti direttamente in C++ da TFLite/MediaPipe,
    che non sono intercettabili tramite variabili d'ambiente Python.
    Chiamare PRIMA di importare mediapipe nel processo worker.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)


# def _normalize_landmarks(landmark_list: list) -> list:
#     """Normalizza i landmark rispetto al polso (Landmark 0)."""
#     base_x, base_y = landmark_list[0][0], landmark_list[0][1]
#     shifted = []
#     for p in landmark_list:
#         shifted.extend([p[0] - base_x, p[1] - base_y])
#     max_val = max(map(abs, shifted))
#     if max_val == 0:
#         return shifted
#     return [v / max_val for v in shifted]

def _normalize_landmarks(landmark_list: list) -> list:
    """Normalizza i landmark 3D (X, Y, Z) rispetto al polso."""
    # landmark_list[0] è il polso: [x, y, z]
    base_x, base_y, base_z = landmark_list[0][0], landmark_list[0][1], landmark_list[0][2]

    shifted = []
    for p in landmark_list:
        # Sottraiamo l'origine (polso) da X, Y e Z
        shifted.extend([
            p[0] - base_x,
            p[1] - base_y,
            p[2] - base_z
        ])

    # Normalizzazione per la scala (divisione per il valore massimo assoluto)
    max_val = max(map(abs, shifted))
    if max_val == 0:
        return shifted

    return [v / max_val for v in shifted]


def _process_chunk(args: tuple) -> list:
    """
    Elabora un chunk di immagini in un processo worker.
    Restituisce una lista di righe CSV (label + 42 coordinate).

    - stderr viene silenziato prima di importare MediaPipe
    - MediaPipe viene istanziato una volta per worker, non per immagine
    - model_complexity=0 (lite) è ~2x più veloce del modello full
    """
    # 1. Sopprimi stderr PRIMA di importare mediapipe (log C++ di TFLite)
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    _suppress_stderr()

    # 2. Import mediapipe solo nel worker, dopo la soppressione
    import mediapipe as mp
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    img_paths, label = args

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        model_complexity=0   # 0=lite (~2x più veloce), 1=full
    )

    rows = []
    try:
        for img_path in img_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Evita una copia interna non necessaria
            rgb_img.flags.writeable = False
            results = hands.process(rgb_img)
            rgb_img.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # raw_points = [
                    #     [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
                    #     for lm in hand_lms.landmark
                    # ]
                    raw_points = [
                        [
                            min(int(lm.x * w), w - 1),
                            min(int(lm.y * h), h - 1),
                            lm.z  # <--- Aggiungiamo Z (non scalata dai pixel, è già relativa)
                        ]
                        for lm in hand_lms.landmark
                    ]
                    normalized = _normalize_landmarks(raw_points)
                    rows.append([label, *normalized])
    finally:
        hands.close()

    return rows


def _chunked(lst: list, n: int):
    """Divide una lista in sotto-liste di dimensione n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class DatasetBuilder:

    def split_images(self) -> bool:
        """Divide fisicamente le immagini in Train / Val / Test."""
        if not os.path.exists(INPUT_FOLDER):
            print(f"[ERROR] Directory '{INPUT_FOLDER}' non trovata.")
            return False

        print(f"\n[INFO] Split dataset da '{INPUT_FOLDER}'...")
        t0 = time.time()
        splitfolders.ratio(INPUT_FOLDER, output=OUTPUT_FOLDER,
                           seed=1337, ratio=(.8, .1, .1), move=False)
        print(f"[OK] Split completato in {time.time() - t0:.2f}s → '{OUTPUT_FOLDER}'")
        return True

    def generate_csv_from_split(self, split_name: str) -> None:
        """
        Legge le immagini dello split e genera il CSV corrispondente.
        Usa un pool di processi per parallelizzare l'estrazione landmark.
        """
        split_path    = Path(OUTPUT_FOLDER) / split_name
        csv_file_path = Path(CSV_DIR) / f"{split_name}_keypoint.csv"

        if not split_path.exists():
            print(f"[WARN] Cartella '{split_path}' non trovata. Skip.")
            return

        print(f"\n[INFO] Generazione CSV per '{split_name}' → {csv_file_path}")
        t_split = time.time()
        Path(CSV_DIR).mkdir(parents=True, exist_ok=True)

        total_images   = 0
        total_detected = 0

        with open(csv_file_path, 'w', newline="", buffering=1 << 20) as f:  # buffer 1 MB
            writer = csv.writer(f)
            buf = []  # buffer righe in RAM

            for folder in sorted(split_path.iterdir(), key=lambda x: int(x.name.split('_')[0])):
                if not folder.is_dir():
                    continue

                # Estrae l'indice numerico dal nome (es. "0_Open" → 0)
                try:
                    label = int(folder.name.split('_')[0])
                except ValueError:
                    print(f"  [WARN] Nome cartella non valido: {folder.name}. Skip.")
                    continue

                # Raccoglie tutti i path immagine validi
                img_paths = [
                    str(p) for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                ]
                num_images = len(img_paths)
                print(f"  → Classe {label} ({folder.name}): {num_images} immagini")

                if num_images == 0:
                    continue

                chunks = list(_chunked(img_paths, CHUNK_SIZE))
                args   = [(chunk, label) for chunk in chunks]

                detected_in_class = 0
                t_class = time.time()

                # Pool di processi: ogni worker ha il proprio modello MediaPipe
                with python_mp.Pool(processes=NUM_WORKERS) as pool:
                    for rows in pool.imap_unordered(_process_chunk, args, chunksize=1):
                        buf.extend(rows)
                        detected_in_class += len(rows)

                        # Flush periodico per non saturare la RAM
                        if len(buf) >= CSV_FLUSH_EVERY:
                            writer.writerows(buf)
                            buf.clear()

                # Flush residui della classe corrente
                if buf:
                    writer.writerows(buf)
                    buf.clear()

                rate = detected_in_class / num_images * 100 if num_images else 0
                print(f"     [OK] {time.time() - t_class:.2f}s | "
                      f"Rilevati: {detected_in_class}/{num_images} ({rate:.1f}%)")

                total_images   += num_images
                total_detected += detected_in_class

        elapsed = time.time() - t_split
        rate    = total_detected / total_images * 100 if total_images else 0
        print(f"[SUMMARY] '{split_name}' → {elapsed:.2f}s | "
              f"{total_detected}/{total_images} hands salvate ({rate:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    python_mp.set_start_method("spawn", force=True)

    t0 = time.time()
    print("=" * 60)
    print(" DATASET BUILDER  ")
    print("=" * 60)
    print(f" Workers  : {NUM_WORKERS}")
    print(f" Chunk    : {CHUNK_SIZE} img/task")
    print(f" Flush    : ogni {CSV_FLUSH_EVERY} righe")
    print("=" * 60)

    builder = DatasetBuilder()

    if builder.split_images():
        for split in ['train', 'val', 'test']:
            builder.generate_csv_from_split(split)

        elapsed = time.time() - t0
        print("\n" + "=" * 60)
        print(" COMPLETATO ")
        print("=" * 60)
        print(f" Tempo totale : {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        print(f" Immagini in  : {OUTPUT_FOLDER}")
        print(f" CSV in       : {CSV_DIR}")
        print("=" * 60)
    else:
        print("\n[ERROR] Processo interrotto: cartella input mancante.")