import os
import cv2
import csv
import mediapipe as mp
import numpy as np
import splitfolders
import time
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURAZIONE PATH ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(project_root, 'dataset', 'hagrid_dataset_512')
OUTPUT_FOLDER = os.path.join(project_root, 'dataset', 'hagrid_dataset_splitted')
CSV_DIR = os.path.join(project_root, "dataset")

# --- PARAMETRI OTTIMIZZATI ---
NUM_WORKERS = 10  # Lascia 2 core liberi per il sistema (i5-12500H ha 12 core)
MAX_HANDS_PER_IMAGE = 2  # Cerca fino a 2 mani per immagine originale


def process_single_image(task_data):
    """
    Processa un'immagine. È globale per compatibilità con il multiprocessing.
    Restituisce una lista di righe (una riga per ogni singola mano trovata).
    """
    img_path, label = task_data
    rows_extracted = []

    # Inizializzazione locale necessaria per isolare i processi in CPU
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=MAX_HANDS_PER_IMAGE,
            min_detection_confidence=0.5
    ) as hands:

        image = cv2.imread(img_path)
        if image is None:
            return rows_extracted

        h, w, _ = image.shape
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Estrazione coordinate (moltiplicate per w, h)
                raw_points = np.array([[lm.x * w, lm.y * h] for lm in hand_lms.landmark])

                # Normalizzazione relativa al polso (landmark 0)
                base_point = raw_points[0]
                shifted = (raw_points - base_point).flatten()

                # Scala i valori tra -1 e 1
                max_val = np.abs(shifted).max()
                if max_val > 0:
                    shifted /= max_val

                # Crea una riga INDIPENDENTE per questa mano: [label, x1, y1, ..., x21, y21]
                row = [label] + shifted.tolist()
                rows_extracted.append(row)

    return rows_extracted

class DatasetBuilder:
    def split_images(self):
        """Divide fisicamente le immagini in Train, Val e Test."""
        if not os.path.exists(INPUT_FOLDER):
            print(f"[ERRORE] Cartella input '{INPUT_FOLDER}' non trovata.")
            return False

        if True:
            return True

        print(f"\n[INFO] Avvio split del dataset da '{INPUT_FOLDER}'...")
        start_time = time.time()

        # Nota: move=False copia i file. Se vuoi svuotare la cartella originale e velocizzare, metti move=True
        splitfolders.ratio(INPUT_FOLDER, output=OUTPUT_FOLDER,
                           seed=1337, ratio=(.8, .1, .1), move=False)

        print(f"[OK] Split completato in {time.time() - start_time:.2f} secondi.")
        return True

    def generate_csv_parallel(self, split_name):
        """Genera il CSV elaborando le immagini contemporaneamente su più core."""
        split_path = os.path.join(OUTPUT_FOLDER, split_name)
        csv_file_path = os.path.join(CSV_DIR, f"{split_name}_keypoint.csv")

        if not os.path.exists(split_path):
            print(f"[WARN] Cartella '{split_path}' non trovata. Salto elaborazione.")
            return

        print(f"\n[INFO] Generazione CSV per '{split_name}' in parallelo...")
        start_time = time.time()
        os.makedirs(CSV_DIR, exist_ok=True)

        # 1. Mappatura dei Task
        tasks = []
        for folder in sorted(os.listdir(split_path)):
            folder_path = os.path.join(split_path, folder)
            if not os.path.isdir(folder_path): continue

            try:
                label = int(folder.split('_')[0])
            except ValueError:
                continue

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isfile(img_path):
                    tasks.append((img_path, label))

        total_images = len(tasks)
        print(f"       Trovate {total_images} immagini. Elaborazione su {NUM_WORKERS} core attivati...")

        # 2. Esecuzione Multiprocessing
        all_rows = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # chunksize=100 spedisce le immagini a blocchi, riducendo i rallentamenti I/O
            for result_rows in executor.map(process_single_image, tasks, chunksize=100):
                if result_rows:
                    all_rows.extend(result_rows)  # Aggiunge tutte le mani trovate (1 o 2) alla lista totale

        # 3. Scrittura Massiva del CSV
        with open(csv_file_path, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(all_rows)

        elapsed = time.time() - start_time
        print(f"[OK] '{split_name}' completato in {elapsed:.2f}s | Mani valide salvate: {len(all_rows)}")


if __name__ == "__main__":
    global_start_time = time.time()

    print("=" * 60)
    print(" AVVIO DATASET BUILDER  ")
    print("=" * 60)

    builder = DatasetBuilder()

    if builder.split_images():
        # Processa i folder generati da splitfolders in sequenza,
        # ma i contenuti interni verranno analizzati in parallelo.
        for split in ['train', 'val', 'test']:
            builder.generate_csv_parallel(split)

        global_elapsed_time = time.time() - global_start_time
        print("\n" + "=" * 60)
        print(" PROCESSO COMPLETATO CON SUCCESSO ")
        print("=" * 60)
        print(f" Tempo totale    : {global_elapsed_time:.2f} secondi ({global_elapsed_time / 60:.2f} minuti)")
        print(f" CSV generati in : {CSV_DIR}")
        print("=" * 60)