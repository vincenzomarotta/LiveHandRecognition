import os
import time
import pandas as pd
import numpy as np
import torch
from statsmodels.stats.contingency_tables import mcnemar

# Import dei modelli per calcolare la latenza "dal vivo"
from nn_model.nn_model import GestureClassifierNN
from cnn_model.cnn_model import GestureClassifierCNN
from heuristic_model.heuristic_model import HeuristicClassifier


def perform_master_analysis():
    print("=" * 75)
    print(" MASTER SCRIPT: ANALISI PAIRWISE (CNN vs NN vs EURISTICO) + EFFICIENZA")
    print("=" * 75)

    project_root = os.path.dirname(os.path.abspath(__file__))
    CSV_H2H = os.path.join(project_root, "head_to_head_results.csv")
    NN_WEIGHTS = os.path.join(project_root, "nn_model", "gesture_nn_model.pth")
    CNN_WEIGHTS = os.path.join(project_root, "cnn_model", "gesture_cnn_finetuned.pth")
    OUTPUT_CSV = os.path.join(project_root, "sommario_risultati_tesi_pairwise.csv")

    final_results = []

    # -----------------------------------------------------------------
    # PARTE 1: ANALISI STATISTICA PAIRWISE (McNemar)
    # -----------------------------------------------------------------
    print("\n[1/3] Lettura dati e Test di McNemar a Coppie...")
    try:
        df = pd.read_csv(CSV_H2H)
    except FileNotFoundError:
        print(f"[ERRORE] File non trovato: {CSV_H2H}")
        print("Esegui test_head_to_head_3way.py prima di questo script.")
        return

    # Usiamo il test Filtrato (solo quando MediaPipe trova la mano)
    # per avere un confronto leale sulle logiche interne dei 3 modelli
    print("Nota: Il confronto statistico viene effettuato sui casi in cui la mano è visibile.")

    # 1. Scontro: CNN vs NN
    final_results.extend(analyze_pairwise(df, "CNN", "cnn_correct", "NN", "nn_correct", filter_mp_failures=True))

    # 2. Scontro: CNN vs Euristico
    final_results.extend(
        analyze_pairwise(df, "CNN", "cnn_correct", "Euristico", "euristico_correct", filter_mp_failures=True))

    # 3. Scontro: NN vs Euristico
    final_results.extend(
        analyze_pairwise(df, "NN", "nn_correct", "Euristico", "euristico_correct", filter_mp_failures=True))

    # -----------------------------------------------------------------
    # PARTE 2: EFFICIENZA SPAZIALE - PESO SUL DISCO (MB)
    # -----------------------------------------------------------------
    print("\n[2/3] Calcolo dimensione modelli su disco...")

    size_mb_nn = os.path.getsize(NN_WEIGHTS) / (1024 * 1024) if os.path.exists(NN_WEIGHTS) else 0
    size_mb_cnn = os.path.getsize(CNN_WEIGHTS) / (1024 * 1024) if os.path.exists(CNN_WEIGHTS) else 0
    size_mb_heur = 0.0  # L'euristico è solo codice, non ha un file .pth di pesi

    print(f"  -> Euristico: {size_mb_heur:.4f} MB (Solo Codice Sorgente)")
    print(f"  -> NN       : {size_mb_nn:.4f} MB")
    print(f"  -> CNN      : {size_mb_cnn:.4f} MB")

    final_results.extend([
        {"Categoria": "Efficienza Spaziale (Memoria)", "Metrica": "Peso Modello Euristico",
         "Valore": f"{size_mb_heur:.4f} MB", "Note": "Nessun parametro appreso"},
        {"Categoria": "Efficienza Spaziale (Memoria)", "Metrica": "Peso Modello NN", "Valore": f"{size_mb_nn:.4f} MB",
         "Note": "Rete 1D su Keypoints"},
        {"Categoria": "Efficienza Spaziale (Memoria)", "Metrica": "Peso Modello CNN", "Valore": f"{size_mb_cnn:.4f} MB",
         "Note": "MobileNetV3 Fine-Tuned"}
    ])

    # -----------------------------------------------------------------
    # PARTE 3: EFFICIENZA TEMPORALE - LATENZA DI INFERENZA (ms)
    # -----------------------------------------------------------------
    print("\n[3/3] Benchmark Latenza (Inferenza a batch=1)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_runs = 500
    warmup_runs = 50

    # Latenza Euristico
    print("  -> Benchmark Euristico in corso...")
    heur_model = HeuristicClassifier((1920, 1080))
    # Creiamo 21 landmark dummy (x, y, z)
    dummy_input_heur = [np.array([np.random.randint(0, 100), np.random.randint(0, 100), np.random.rand()]) for _ in
                        range(21)]

    for _ in range(warmup_runs): _ = heur_model.process(dummy_input_heur)

    latencies_heur = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = heur_model.process(dummy_input_heur)
        latencies_heur.append((time.perf_counter() - t0) * 1000)
    avg_lat_heur = np.mean(latencies_heur)
    print(f"  -> Latenza Media Euristico : {avg_lat_heur:.3f} ms")

    # Latenza NN
    print("  -> Benchmark NN in corso...")
    nn_model = GestureClassifierNN(input_dim=73, num_classes=15)
    dummy_input_nn = [np.random.rand() for _ in range(73)]
    for _ in range(warmup_runs): _ = nn_model.predict(dummy_input_nn)

    latencies_nn = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = nn_model.predict(dummy_input_nn)
        latencies_nn.append((time.perf_counter() - t0) * 1000)
    avg_lat_nn = np.mean(latencies_nn)
    print(f"  -> Latenza Media NN        : {avg_lat_nn:.3f} ms")

    # Latenza CNN
    print("  -> Benchmark CNN in corso...")
    cnn_model = GestureClassifierCNN(num_classes=15)
    cnn_model.model.eval()
    dummy_input_cnn = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        for _ in range(warmup_runs):
            _ = cnn_model.model(dummy_input_cnn)
            if torch.cuda.is_available(): torch.cuda.synchronize()

        latencies_cnn = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = cnn_model.model(dummy_input_cnn)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            latencies_cnn.append((time.perf_counter() - t0) * 1000)

    avg_lat_cnn = np.mean(latencies_cnn)
    print(f"  -> Latenza Media CNN       : {avg_lat_cnn:.3f} ms")

    final_results.extend([
        {"Categoria": "Efficienza Temporale (Velocità)", "Metrica": "Latenza Media Euristico",
         "Valore": f"{avg_lat_heur:.3f} ms", "Note": "Esecuzione regole if-else (CPU)"},
        {"Categoria": "Efficienza Temporale (Velocità)", "Metrica": "Latenza Media NN",
         "Valore": f"{avg_lat_nn:.3f} ms", "Note": "Tempo elaborazione rete (CPU/GPU)"},
        {"Categoria": "Efficienza Temporale (Velocità)", "Metrica": "Latenza Media CNN",
         "Valore": f"{avg_lat_cnn:.3f} ms", "Note": "Tempo elaborazione End-to-End (GPU)"}
    ])

    # -----------------------------------------------------------------
    # ESPORTAZIONE CSV
    # -----------------------------------------------------------------
    print(f"\nSalvataggio report su CSV: {OUTPUT_CSV}...")
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("[OK] Master CSV generato con successo! Pronto per la tesi.")
    print("=" * 75)


def analyze_pairwise(df, name_a, col_a, name_b, col_b, filter_mp_failures):
    results = []

    if filter_mp_failures:
        data = df[df['mediapipe_failed'] == 0]
        desc = "(Su Mani Rilevate)"
    else:
        data = df
        desc = "(Inclusi Fallimenti)"

    total = len(data)

    # Preveniamo errori se le colonne mancano
    if col_a not in data.columns or col_b not in data.columns:
        print(f"[ERRORE] Colonne {col_a} o {col_b} mancanti nel CSV.")
        return results

    vec_a = data[col_a].astype(bool)
    vec_b = data[col_b].astype(bool)

    # Tabella di contingenza
    a = np.sum(vec_a & vec_b)
    b = np.sum(vec_a & ~vec_b)  # Solo Modello A ha indovinato
    c = np.sum(~vec_a & vec_b)  # Solo Modello B ha indovinato
    d = np.sum(~vec_a & ~vec_b)

    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=False, correction=True)

    prefix = f"Scontro: {name_a} vs {name_b} {desc}"

    # Accuratezze specifiche di questo set di dati
    acc_a = (a + b) / total * 100
    acc_b = (a + c) / total * 100

    if b > c:
        winner = name_a
    elif c > b:
        winner = name_b
    else:
        winner = "Pareggio Assoluto"

    is_significant = result.pvalue < 0.05
    esito = f"Vince {winner}" if is_significant else "Equivalenti (Non Significativo)"

    results.extend([
        {"Categoria": prefix, "Metrica": f"Accuratezza {name_a}", "Valore": f"{acc_a:.2f}%", "Note": ""},
        {"Categoria": prefix, "Metrica": f"Accuratezza {name_b}", "Valore": f"{acc_b:.2f}%", "Note": ""},
        {"Categoria": prefix, "Metrica": f"Solo {name_a} Corretto (b)", "Valore": str(b),
         "Note": f"Disaccordo pro {name_a}"},
        {"Categoria": prefix, "Metrica": f"Solo {name_b} Corretto (c)", "Valore": str(c),
         "Note": f"Disaccordo pro {name_b}"},
        {"Categoria": prefix, "Metrica": "McNemar: P-Value", "Valore": f"{result.pvalue:.10f}",
         "Note": "< 0.05 significa divario reale"},
        {"Categoria": prefix, "Metrica": "Esito Scontro", "Valore": esito, "Note": ""}
    ])

    return results


if __name__ == "__main__":
    perform_master_analysis()