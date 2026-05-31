"""
Recall @ fixed False-Positive-Rate (FPR) — stessa metrica del paper Google 2021
("On-device Real-time Hand Gesture Recognition": NN 87.9% recall @ 1% FPR,
 euristico 44.4% recall @ 0.86% FPR).

COSA FA
-------
Per ogni classe-gesto (one-vs-rest) si sceglie una soglia sullo score del modello
tale che il tasso di falsi positivi sui campioni NEGATIVI sia pari a un target
(es. 1%); a quella soglia si misura il recall (TPR). Si fa la media sulle classi-gesto.

- NN e CNN producono PROBABILITA' continue -> si può spazzolare la soglia.
- L'euristico produce solo etichette dure (nessuno score) -> ha UN solo punto di
  lavoro: si misura il suo FPR reale e il suo recall reale (come nel paper).

DEFINIZIONE NEGATIVI
--------------------
NEGATIVE_MODE = 'background' : i negativi sono i campioni NO GESTURE (label 14),
                               come la classe "Negative/background" del paper.
NEGATIVE_MODE = 'rest'       : one-vs-rest puro (i negativi sono tutte le altre classi).

ESECUZIONE
----------
1) Pass di inferenza sul test set (richiede GPU + dataset immagini + pesi .pth).
   Gli score vengono salvati in utils/score_dump.csv.
2) Calcolo recall@FPR dagli score salvati -> utils/recall_at_fpr_results.csv.

Se score_dump.csv esiste già e REUSE_DUMP=True, salta la (1): puoi ritarare i
target di FPR senza rifare l'inferenza.

    python utils/recall_at_fpr.py
"""

import os
import csv
from pathlib import Path

import numpy as np

# ===========================================================================
# CONFIG
# ===========================================================================
ROOT = Path(__file__).resolve().parents[1]          # cartella progetto (parent di utils/)
TEST_DIR = ROOT / "dataset" / "hagrid_dataset_split" / "test"
NN_WEIGHTS = ROOT / "nn_model" / "gesture_nn_model.pth"
CNN_WEIGHTS = ROOT / "cnn_model" / "gesture_cnn_finetuned.pth"

DUMP_CSV = Path(__file__).resolve().parent / "score_dump.csv"
RESULTS_CSV = Path(__file__).resolve().parent / "recall_at_fpr_results.csv"

GESTURE_NAMES = [
    'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
    'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP',
    'THREE', 'TWO', 'NO GESTURE'
]
NUM_CLASSES = 15
BACKGROUND_LABEL = 14                 # NO GESTURE
GESTURE_LABELS = [i for i in range(NUM_CLASSES) if i != BACKGROUND_LABEL]

# --- parametri analisi ---
TARGET_FPRS = [0.01, 0.005, 0.02]     # 1% = punto del paper
NEGATIVE_MODE = 'background'          # 'background' (come il paper) | 'rest'
USE_DETECTED_ONLY = True             # solo immagini dove MediaPipe trova la mano
                                      # (confronto leale NN/CNN/euristico)
CNN_ON_ALL = False                    # se True, la CNN usa anche le immagini
                                      # senza mano rilevata (suo vantaggio nativo)
REUSE_DUMP = True                     # riusa score_dump.csv se presente


# ===========================================================================
# FASE 1 — INFERENZA (produce score_dump.csv)
# ===========================================================================
def run_inference_and_dump():
    import cv2
    import torch
    from torchvision import transforms
    from PIL import Image
    import mediapipe as mp

    from nn_model.nn_model import GestureClassifierNN
    from cnn_model.cnn_model import GestureClassifierCNN
    from heuristic_model.heuristic_model import HeuristicClassifier
    from utils.utility import (pre_process_landmark, normalize_wrt_keypoint,
                               calculate_finger_angles)

    if not TEST_DIR.is_dir():
        raise SystemExit(f"[ERRORE] Test set non trovato: {TEST_DIR}")

    print("Caricamento modelli...")
    nn_clf = GestureClassifierNN(input_dim=73, num_classes=NUM_CLASSES)
    nn_clf.load(str(NN_WEIGHTS))

    cnn_clf = GestureClassifierCNN(num_classes=NUM_CLASSES)
    cnn_clf.model.load_state_dict(torch.load(str(CNN_WEIGHTS), map_location=cnn_clf.device))
    cnn_clf.model.eval()

    heur_clf = HeuristicClassifier((1920, 1080))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                           min_detection_confidence=0.5, model_complexity=0)

    cnn_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ImageFolder ordina alfabeticamente -> indice CNN != label reale.
    # Costruisco cnn_idx -> true_label per RIORDINARE il vettore di probabilità
    # in ordine di label reale.
    alpha = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
    cnn_to_true = {idx: int(name.split('_')[0]) for idx, name in enumerate(alpha)}

    def softmax_np(logits_tensor):
        return torch.softmax(logits_tensor, dim=1).squeeze(0).detach().cpu().numpy()

    rows = []
    total, mp_fail = 0, 0

    for folder in sorted(TEST_DIR.iterdir(), key=lambda x: int(x.name.split('_')[0])):
        if not folder.is_dir():
            continue
        true_label = int(folder.name.split('_')[0])
        imgs = [p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        print(f"Classe {true_label:2d} ({folder.name}): {len(imgs)} immagini...")

        for img_path in imgs:
            try:
                pil = Image.open(str(img_path)).convert('RGB')
            except Exception:
                continue
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                continue
            total += 1
            fh, fw, _ = cv_img.shape

            # --- CNN (probabilità riordinate in ordine di label reale) ---
            t = cnn_tf(pil).unsqueeze(0).to(cnn_clf.device)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                cnn_raw = softmax_np(cnn_clf.model(t).float())
            cnn_probs = np.zeros(NUM_CLASSES, dtype=np.float64)
            for cnn_idx, p in enumerate(cnn_raw):
                cnn_probs[cnn_to_true[cnn_idx]] = p

            # --- NN + euristico (servono i landmark MediaPipe) ---
            nn_probs = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
            heur_pred = -1
            res = hands.process(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]

                pure = [[lm.x, lm.y, lm.z] for lm in hl.landmark]
                angles = calculate_finger_angles(pure)
                lm2d = pre_process_landmark(hl, cv_img)
                norm = normalize_wrt_keypoint(lm2d, lm2d[0], one_d=True)
                feats = norm + angles
                with torch.no_grad():
                    logits = nn_clf.model(torch.tensor([feats], dtype=torch.float32).to(nn_clf.device))
                nn_probs = softmax_np(logits.float())

                heur_lm = [np.array([min(int(lm.x * fw), fw - 1),
                                     min(int(lm.y * fh), fh - 1), lm.z], dtype=np.float64)
                           for lm in hl.landmark]
                heur_clf.frame_width = fw
                heur_clf.frame_height = fh
                heur_pred = heur_clf.process(heur_lm)
            else:
                mp_fail += 1

            row = {'img_name': img_path.name, 'true_label': true_label,
                   'heur_pred': heur_pred,
                   'mediapipe_failed': 1 if heur_pred == -1 else 0}
            for k in range(NUM_CLASSES):
                row[f'nn_p{k}'] = nn_probs[k]
                row[f'cnn_p{k}'] = cnn_probs[k]
            rows.append(row)

    hands.close()

    with open(DUMP_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {len(rows)} righe salvate in {DUMP_CSV} "
          f"(MediaPipe ha fallito su {mp_fail}).")
    return rows


# ===========================================================================
# FASE 2 — CALCOLO RECALL @ FPR (dagli score salvati)
# ===========================================================================
def load_dump():
    rows = []
    with open(DUMP_CSV, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    n = len(rows)
    true = np.array([int(r['true_label']) for r in rows])
    mp_failed = np.array([int(r['mediapipe_failed']) for r in rows])
    heur = np.array([int(r['heur_pred']) for r in rows])
    nn = np.full((n, NUM_CLASSES), np.nan)
    cnn = np.full((n, NUM_CLASSES), np.nan)
    for i, r in enumerate(rows):
        for k in range(NUM_CLASSES):
            nn[i, k] = float(r[f'nn_p{k}'])
            cnn[i, k] = float(r[f'cnn_p{k}'])
    return true, mp_failed, heur, nn, cnn


def threshold_at_fpr(neg_scores, target_fpr):
    """Soglia t tale che frac(neg >= t) == target_fpr (regola >=).
    Con pochi negativi l'FPR è quantizzato: si prende la soglia più bassa
    che NON supera il target."""
    s = np.sort(neg_scores)[::-1]            # decrescente
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.floor(target_fpr * n))        # quanti FP ci concediamo
    if k <= 0:
        # nessun FP ammesso: soglia appena sopra il massimo negativo
        return np.nextafter(s[0], np.inf)
    if k >= n:
        return -np.inf
    return s[k - 1]                          # i primi k negativi sono FP


def recall_at_fpr_scorebased(scores, true, target_fpr, mode):
    """Ritorna (recall medio, dict per-classe) per un modello con score continui."""
    per_class = {}
    recalls = []
    for k in GESTURE_LABELS:
        pos = scores[true == k, k]
        if mode == 'background':
            neg = scores[true == BACKGROUND_LABEL, k]
        else:  # 'rest'
            neg = scores[(true != k) & (true != BACKGROUND_LABEL), k]
            neg = np.concatenate([neg, scores[true == BACKGROUND_LABEL, k]])
        if len(pos) == 0:
            continue
        t = threshold_at_fpr(neg, target_fpr)
        rec = float(np.mean(pos >= t))
        per_class[GESTURE_NAMES[k]] = rec
        recalls.append(rec)
    return float(np.mean(recalls)) if recalls else float('nan'), per_class


def heuristic_operating_point(heur, true, mode):
    """L'euristico ha un solo punto: misuro il suo FPR reale e il recall reale."""
    # FPR: frazione di negativi classificati come QUALCHE gesto (pred != background)
    if mode == 'background':
        neg_mask = true == BACKGROUND_LABEL
        fp = np.sum((heur[neg_mask] != BACKGROUND_LABEL) & (heur[neg_mask] != -1))
        fpr = fp / max(np.sum(neg_mask), 1)
    else:
        # one-vs-rest medio del FPR sulle classi-gesto
        fprs = []
        for k in GESTURE_LABELS:
            neg_mask = true != k
            fprs.append(np.mean(heur[neg_mask] == k))
        fpr = float(np.mean(fprs))

    per_class = {}
    recalls = []
    for k in GESTURE_LABELS:
        pos_mask = true == k
        if np.sum(pos_mask) == 0:
            continue
        rec = float(np.mean(heur[pos_mask] == k))
        per_class[GESTURE_NAMES[k]] = rec
        recalls.append(rec)
    return (float(np.mean(recalls)) if recalls else float('nan')), float(fpr), per_class


def analyze():
    true, mp_failed, heur, nn, cnn = load_dump()

    # sottoinsieme di confronto
    mask = np.ones(len(true), dtype=bool)
    if USE_DETECTED_ONLY:
        mask = mp_failed == 0
    true_s, heur_s, nn_s = true[mask], heur[mask], nn[mask]
    cnn_s = cnn if CNN_ON_ALL else cnn[mask]
    cnn_true = true if CNN_ON_ALL else true_s

    print("\n" + "=" * 70)
    print(f" RECALL @ FPR  |  negativi='{NEGATIVE_MODE}'  |  "
          f"subset={'mani rilevate' if USE_DETECTED_ONLY else 'tutte'}")
    print("=" * 70)
    n_bg = int(np.sum(true_s == BACKGROUND_LABEL))
    print(f"Campioni: {int(np.sum(mask))} (di cui {n_bg} NO GESTURE come negativi)")
    if n_bg < 200 and NEGATIVE_MODE == 'background':
        print("[!] Pochi negativi: l'FPR è quantizzato, i valori vanno letti con cautela.")

    out_rows = []

    # --- euristico: punto singolo ---
    h_rec, h_fpr, h_pc = heuristic_operating_point(heur_s, true_s, NEGATIVE_MODE)
    print(f"\nEURISTICO (punto di lavoro fisso): "
          f"recall medio = {h_rec*100:.2f}%  @ FPR reale = {h_fpr*100:.2f}%")
    out_rows.append({'modello': 'Euristico', 'target_fpr': f'{h_fpr*100:.2f}% (reale)',
                     'recall_medio': f'{h_rec*100:.2f}%'})

    # --- NN e CNN: sweep su target FPR ---
    for target in TARGET_FPRS:
        nn_rec, nn_pc = recall_at_fpr_scorebased(nn_s, true_s, target, NEGATIVE_MODE)
        cnn_rec, cnn_pc = recall_at_fpr_scorebased(cnn_s, cnn_true, target, NEGATIVE_MODE)
        print(f"\n@ FPR = {target*100:.2f}%")
        print(f"   NN  : recall medio = {nn_rec*100:.2f}%")
        print(f"   CNN : recall medio = {cnn_rec*100:.2f}%")
        out_rows.append({'modello': 'NN', 'target_fpr': f'{target*100:.2f}%',
                         'recall_medio': f'{nn_rec*100:.2f}%'})
        out_rows.append({'modello': 'CNN', 'target_fpr': f'{target*100:.2f}%',
                         'recall_medio': f'{cnn_rec*100:.2f}%'})

    # dettaglio per-classe @ 1% (o primo target) per NN/CNN ed euristico
    t0 = TARGET_FPRS[0]
    _, nn_pc = recall_at_fpr_scorebased(nn_s, true_s, t0, NEGATIVE_MODE)
    _, cnn_pc = recall_at_fpr_scorebased(cnn_s, cnn_true, t0, NEGATIVE_MODE)
    print(f"\n--- Recall per-classe (NN/CNN @ {t0*100:.2f}% FPR, Euristico al suo punto) ---")
    print(f"{'gesto':<14}{'NN':>10}{'CNN':>10}{'Eur':>10}")
    for k in GESTURE_LABELS:
        name = GESTURE_NAMES[k]
        print(f"{name:<14}{nn_pc.get(name,0)*100:>9.1f}%{cnn_pc.get(name,0)*100:>9.1f}%"
              f"{h_pc.get(name,0)*100:>9.1f}%")

    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['modello', 'target_fpr', 'recall_medio'])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[OK] Riepilogo salvato in {RESULTS_CSV}")


# ===========================================================================
if __name__ == "__main__":
    if not (REUSE_DUMP and DUMP_CSV.exists()):
        run_inference_and_dump()
    else:
        print(f"[i] Riuso {DUMP_CSV} (REUSE_DUMP=True). "
              f"Cancellalo per rifare l'inferenza.")
    analyze()
