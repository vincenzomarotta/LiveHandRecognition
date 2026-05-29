import os
import time
import pandas as pd
import numpy as np
import torch
from statsmodels.stats.contingency_tables import mcnemar

# Import dei tuoi modelli
from nn_model.nn_model import GestureClassifierNN
from cnn_model.cnn_model import GestureClassifierCNN


def perform_master_analysis():
    print("=" * 70)
    print(" MASTER SCRIPT: ANALISI COMPLETA PER TESI (EFFICACIA + EFFICIENZA)")
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))
    CSV_H2H = os.path.join(project_root, "utils", "head_to_head_results.csv")
    NN_WEIGHTS = os.path.join(project_root, "nn_model", "gesture_nn_model.pth")
    CNN_WEIGHTS = os.path.join(project_root, "cnn_model", "gesture_cnn_finetuned.pth")
    OUTPUT_CSV = os.path.join(project_root, "utils", "sommario_risultati_tesi_reale.csv")

    # Dizionario che conterrà tutti i risultati da salvare
    final_results = []

    # -----------------------------------------------------------------
    # PARTE 1: ANALISI STATISTICA (McNemar) dal file Head-to-Head
    # -----------------------------------------------------------------
    print("\n[1/3] Lettura dati e Test di McNemar...")
    try:
        df = pd.read_csv(CSV_H2H)
    except FileNotFoundError:
        print(f"[ERRORE] File Head-to-Head non trovato: {CSV_H2H}")
        print("Assicurati di aver eseguito test_head_to_head.py prima di questo script.")
        return

    # Scenario A (Tutti i dati)
    final_results.extend(analyze_scenario(df, "A", filter_mp_failures=False))

    # Scenario B (Solo mano visibile)
    final_results.extend(analyze_scenario(df, "B", filter_mp_failures=True))

    # -----------------------------------------------------------------
    # PARTE 2: SOSTENIBILITA' - PESO SUL DISCO (Footprint)
    # -----------------------------------------------------------------
    print("\n[2/3] Calcolo dimensione modelli su disco...")

    size_mb_nn = os.path.getsize(NN_WEIGHTS) / (1024 * 1024) if os.path.exists(NN_WEIGHTS) else 0
    size_mb_cnn = os.path.getsize(CNN_WEIGHTS) / (1024 * 1024) if os.path.exists(CNN_WEIGHTS) else 0

    print(f"  -> NN: {size_mb_nn:.4f} MB")
    print(f"  -> CNN: {size_mb_cnn:.4f} MB")

    final_results.extend([
        {"Categoria": "Efficienza Spaziale (Memoria)", "Metrica": "Peso File Modello NN",
         "Valore": f"{size_mb_nn:.4f} MB", "Note": "Rete 1D su Keypoints"},
        {"Categoria": "Efficienza Spaziale (Memoria)", "Metrica": "Peso File Modello CNN",
         "Valore": f"{size_mb_cnn:.4f} MB", "Note": "MobileNetV3 Fine-Tuned"},
        {"Categoria": "Efficienza Spaziale (Memoria)", "Metrica": "Rapporto di Grandezza",
         "Valore": f"CNN è {size_mb_cnn / size_mb_nn:.1f}x più pesante", "Note": "Impatto storage"}
    ])

    # -----------------------------------------------------------------
    # PARTE 3: SOSTENIBILITA' - LATENZA DI INFERENZA REALE
    # -----------------------------------------------------------------
    print("\n[3/3] Benchmark Latenza (Inferenza a batch=1)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_runs = 500
    warmup_runs = 50

    # Latenza NN
    print("  -> Scaldamento NN in corso...")
    nn_model = GestureClassifierNN(input_dim=73, num_classes=15)
    dummy_input_nn = [np.random.rand() for _ in range(73)]
    for _ in range(warmup_runs): _ = nn_model.predict(dummy_input_nn)

    latencies_nn = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = nn_model.predict(dummy_input_nn)
        latencies_nn.append((time.perf_counter() - t0) * 1000)
    avg_lat_nn = np.mean(latencies_nn)
    print(f"  -> Latenza Media NN: {avg_lat_nn:.3f} ms")

    # Latenza CNN
    print("  -> Scaldamento CNN in corso...")
    cnn_model = GestureClassifierCNN(num_classes=15)
    cnn_model.model.eval()
    dummy_input_cnn = torch.randn(1, 3, 224, 224).to(device)
    latencies_cnn = []

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        for _ in range(warmup_runs):
            _ = cnn_model.model(dummy_input_cnn)
            if torch.cuda.is_available(): torch.cuda.synchronize()

        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = cnn_model.model(dummy_input_cnn)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            latencies_cnn.append((time.perf_counter() - t0) * 1000)

    avg_lat_cnn = np.mean(latencies_cnn)
    print(f"  -> Latenza Media CNN: {avg_lat_cnn:.3f} ms")

    final_results.extend([
        {"Categoria": "Efficienza Temporale (Velocità)", "Metrica": "Latenza Media NN",
         "Valore": f"{avg_lat_nn:.3f} ms", "Note": "Tempo elaborazione rete (Escluso MediaPipe)"},
        {"Categoria": "Efficienza Temporale (Velocità)", "Metrica": "Latenza Media CNN",
         "Valore": f"{avg_lat_cnn:.3f} ms", "Note": "Tempo elaborazione End-to-End"},
        {"Categoria": "Efficienza Temporale (Velocità)", "Metrica": "Rapporto di Velocità",
         "Valore": f"NN è {avg_lat_cnn / avg_lat_nn:.1f}x più veloce", "Note": "Impatto fluidità/batteria"}
    ])

    # -----------------------------------------------------------------
    # ESPORTAZIONE CSV
    # -----------------------------------------------------------------
    print(f"\nSalvataggio report su CSV: {OUTPUT_CSV}...")
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("[OK] Master CSV generato con successo! Pronto per la tesi.")
    print("=" * 70)


def analyze_scenario(df, scenario_name, filter_mp_failures):
    results = []

    if filter_mp_failures:
        data = df[df['mediapipe_failed'] == 0]
        desc = "(Solo mani rilevate da MediaPipe)"
    else:
        data = df
        desc = "(Inclusi fallimenti MediaPipe)"

    total = len(data)
    cnn_correct = data['cnn_correct'].astype(bool)
    nn_correct = data['nn_correct'].astype(bool)

    a = np.sum(cnn_correct & nn_correct)
    b = np.sum(cnn_correct & ~nn_correct)
    c = np.sum(~cnn_correct & nn_correct)
    d = np.sum(~cnn_correct & ~nn_correct)

    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=False, correction=True)

    prefix = f"Scenario {scenario_name} {desc}"

    # Accuratezze
    acc_cnn = (a + b) / total * 100
    acc_nn = (a + c) / total * 100

    results.extend([
        {"Categoria": prefix, "Metrica": "Campioni Analizzati", "Valore": str(total), "Note": ""},
        {"Categoria": prefix, "Metrica": "Accuratezza CNN", "Valore": f"{acc_cnn:.2f}%", "Note": ""},
        {"Categoria": prefix, "Metrica": "Accuratezza NN", "Valore": f"{acc_nn:.2f}%", "Note": ""},
        {"Categoria": prefix, "Metrica": "McNemar: Entrambi Corretti", "Valore": str(a), "Note": ""},
        {"Categoria": prefix, "Metrica": "McNemar: Solo CNN Corretta", "Valore": str(b), "Note": "Disaccordo pro CNN"},
        {"Categoria": prefix, "Metrica": "McNemar: Solo NN Corretta", "Valore": str(c), "Note": "Disaccordo pro NN"},
        {"Categoria": prefix, "Metrica": "McNemar: Entrambi Errati", "Valore": str(d), "Note": ""},
        {"Categoria": prefix, "Metrica": "McNemar: Chi-Quadro", "Valore": f"{result.statistic:.4f}", "Note": ""},
        {"Categoria": prefix, "Metrica": "McNemar: P-Value", "Valore": f"{result.pvalue:.10f}", "Note": ""},
        {"Categoria": prefix, "Metrica": "McNemar: Esito",
         "Valore": "Differenza Significativa" if result.pvalue < 0.05 else "Differenza Non Significativa",
         "Note": f"Vince {'CNN' if b > c else 'NN'}"}
    ])

    return results


if __name__ == "__main__":
    perform_master_analysis()