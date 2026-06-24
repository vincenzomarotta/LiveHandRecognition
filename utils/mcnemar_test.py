import os
import time
import pandas as pd
import numpy as np
import torch
from statsmodels.stats.contingency_tables import mcnemar

# Import the models to benchmark live inference latency
from nn_model.nn_model import GestureClassifierNN
from cnn_model.cnn_model import GestureClassifierCNN
from heuristic_model.heuristic_model import HeuristicClassifier


def perform_master_analysis():
    print("=" * 75)
    print(" MASTER SCRIPT: PAIRWISE ANALYSIS (CNN vs NN vs HEURISTIC) + EFFICIENCY")
    print("=" * 75)

    project_root = os.path.dirname(os.path.abspath(__file__))
    CSV_H2H = os.path.join(project_root, "head_to_head_results.csv")
    NN_WEIGHTS = os.path.join(project_root, "nn_model", "gesture_nn_model.pth")
    CNN_WEIGHTS = os.path.join(project_root, "cnn_model", "gesture_cnn_finetuned.pth")
    OUTPUT_CSV = os.path.join(project_root, "pairwise_results_summary.csv")

    final_results = []

    # -----------------------------------------------------------------
    # PART 1: PAIRWISE STATISTICAL ANALYSIS (McNemar)
    # -----------------------------------------------------------------
    print("\n[1/3] Reading data and running pairwise McNemar tests...")
    try:
        df = pd.read_csv(CSV_H2H)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {CSV_H2H}")
        print("Run test_head_to_head.py before this script.")
        return

    # The statistical comparison uses only cases where MediaPipe found the hand,
    # for a fair comparison of the three models' internal logic.
    print("Note: the statistical comparison is run on cases where the hand is visible.")

    # 1. CNN vs NN
    final_results.extend(analyze_pairwise(df, "CNN", "cnn_correct", "NN", "nn_correct", filter_mp_failures=True))

    # 2. CNN vs Heuristic
    final_results.extend(
        analyze_pairwise(df, "CNN", "cnn_correct", "Heuristic", "euristico_correct", filter_mp_failures=True))

    # 3. NN vs Heuristic
    final_results.extend(
        analyze_pairwise(df, "NN", "nn_correct", "Heuristic", "euristico_correct", filter_mp_failures=True))

    # -----------------------------------------------------------------
    # PART 2: SPATIAL EFFICIENCY - ON-DISK SIZE (MB)
    # -----------------------------------------------------------------
    print("\n[2/3] Computing on-disk model sizes...")

    size_mb_nn = os.path.getsize(NN_WEIGHTS) / (1024 * 1024) if os.path.exists(NN_WEIGHTS) else 0
    size_mb_cnn = os.path.getsize(CNN_WEIGHTS) / (1024 * 1024) if os.path.exists(CNN_WEIGHTS) else 0
    size_mb_heur = 0.0  # the heuristic is pure code, no .pth weights file

    print(f"  -> Heuristic: {size_mb_heur:.4f} MB (source code only)")
    print(f"  -> NN       : {size_mb_nn:.4f} MB")
    print(f"  -> CNN      : {size_mb_cnn:.4f} MB")

    final_results.extend([
        {"Category": "Spatial Efficiency (Memory)", "Metric": "Heuristic Model Size",
         "Value": f"{size_mb_heur:.4f} MB", "Note": "No learned parameters"},
        {"Category": "Spatial Efficiency (Memory)", "Metric": "NN Model Size", "Value": f"{size_mb_nn:.4f} MB",
         "Note": "1D network on keypoints"},
        {"Category": "Spatial Efficiency (Memory)", "Metric": "CNN Model Size", "Value": f"{size_mb_cnn:.4f} MB",
         "Note": "MobileNetV3 fine-tuned"}
    ])

    # -----------------------------------------------------------------
    # PART 3: TEMPORAL EFFICIENCY - INFERENCE LATENCY (ms)
    # -----------------------------------------------------------------
    print("\n[3/3] Latency benchmark (inference at batch=1)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_runs = 500
    warmup_runs = 50

    # Heuristic latency
    print("  -> Benchmarking heuristic...")
    heur_model = HeuristicClassifier((1920, 1080))
    # 21 dummy landmarks (x, y, z)
    dummy_input_heur = [np.array([np.random.randint(0, 100), np.random.randint(0, 100), np.random.rand()]) for _ in
                        range(21)]

    for _ in range(warmup_runs): _ = heur_model.process(dummy_input_heur)

    latencies_heur = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = heur_model.process(dummy_input_heur)
        latencies_heur.append((time.perf_counter() - t0) * 1000)
    avg_lat_heur = np.mean(latencies_heur)
    print(f"  -> Average heuristic latency : {avg_lat_heur:.3f} ms")

    # NN latency
    print("  -> Benchmarking NN...")
    nn_model = GestureClassifierNN(input_dim=73, num_classes=15)
    dummy_input_nn = [np.random.rand() for _ in range(73)]
    for _ in range(warmup_runs): _ = nn_model.predict(dummy_input_nn)

    latencies_nn = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = nn_model.predict(dummy_input_nn)
        latencies_nn.append((time.perf_counter() - t0) * 1000)
    avg_lat_nn = np.mean(latencies_nn)
    print(f"  -> Average NN latency        : {avg_lat_nn:.3f} ms")

    # CNN latency
    print("  -> Benchmarking CNN...")
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
    print(f"  -> Average CNN latency       : {avg_lat_cnn:.3f} ms")

    final_results.extend([
        {"Category": "Temporal Efficiency (Speed)", "Metric": "Average Heuristic Latency",
         "Value": f"{avg_lat_heur:.3f} ms", "Note": "if-else rule execution (CPU)"},
        {"Category": "Temporal Efficiency (Speed)", "Metric": "Average NN Latency",
         "Value": f"{avg_lat_nn:.3f} ms", "Note": "network processing time (CPU/GPU)"},
        {"Category": "Temporal Efficiency (Speed)", "Metric": "Average CNN Latency",
         "Value": f"{avg_lat_cnn:.3f} ms", "Note": "end-to-end processing time (GPU)"}
    ])

    # -----------------------------------------------------------------
    # CSV EXPORT
    # -----------------------------------------------------------------
    print(f"\nSaving report to CSV: {OUTPUT_CSV}...")
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("[OK] Master CSV generated successfully.")
    print("=" * 75)


def analyze_pairwise(df, name_a, col_a, name_b, col_b, filter_mp_failures):
    results = []

    if filter_mp_failures:
        data = df[df['mediapipe_failed'] == 0]
        desc = "(On Detected Hands)"
    else:
        data = df
        desc = "(Including Failures)"

    total = len(data)

    # Guard against missing columns
    if col_a not in data.columns or col_b not in data.columns:
        print(f"[ERROR] Columns {col_a} or {col_b} missing in the CSV.")
        return results

    vec_a = data[col_a].astype(bool)
    vec_b = data[col_b].astype(bool)

    # Contingency table
    a = np.sum(vec_a & vec_b)
    b = np.sum(vec_a & ~vec_b)  # only model A correct
    c = np.sum(~vec_a & vec_b)  # only model B correct
    d = np.sum(~vec_a & ~vec_b)

    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=False, correction=True)

    prefix = f"Match: {name_a} vs {name_b} {desc}"

    # Accuracy on this dataset
    acc_a = (a + b) / total * 100
    acc_b = (a + c) / total * 100

    if b > c:
        winner = name_a
    elif c > b:
        winner = name_b
    else:
        winner = "Exact Tie"

    is_significant = result.pvalue < 0.05
    outcome = f"{winner} wins" if is_significant else "Equivalent (Not Significant)"

    results.extend([
        {"Category": prefix, "Metric": f"Accuracy {name_a}", "Value": f"{acc_a:.2f}%", "Note": ""},
        {"Category": prefix, "Metric": f"Accuracy {name_b}", "Value": f"{acc_b:.2f}%", "Note": ""},
        {"Category": prefix, "Metric": f"Only {name_a} Correct (b)", "Value": str(b),
         "Note": f"Disagreement favoring {name_a}"},
        {"Category": prefix, "Metric": f"Only {name_b} Correct (c)", "Value": str(c),
         "Note": f"Disagreement favoring {name_b}"},
        {"Category": prefix, "Metric": "McNemar: P-Value", "Value": f"{result.pvalue:.10f}",
         "Note": "< 0.05 means a real gap"},
        {"Category": prefix, "Metric": "Match Outcome", "Value": outcome, "Note": ""}
    ])

    return results


if __name__ == "__main__":
    perform_master_analysis()
