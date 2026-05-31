"""
Genera i grafici della presentazione leggendo i CSV di risultati reali.
Salva i PNG direttamente in assets/ con i nomi usati da index.html.

Run:  python -m utils.presentation_plots.make_plots
  (oppure)  python utils/presentation_plots/make_plots.py
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "assets")           # salva direttamente negli asset del deck

CNN_CSV = os.path.join(ROOT, "cnn_model", "metrics_summary_cnn.csv")
NN_CSV = os.path.join(ROOT, "nn_model", "metrics_summary_nn.csv")
HEUR_CSV = os.path.join(ROOT, "heuristic_model", "metrics_summary_heuristic.csv")
PAIRWISE_CSV = os.path.join(ROOT, "utils", "sommario_risultati_tesi_pairwise.csv")

GESTURE_NAMES = [
    'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
    'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP', 'THREE', 'TWO', 'NO GESTURE'
]

# Palette coerente per i tre modelli
C_HEUR = "#E07A5F"   # terracotta
C_NN = "#3D9970"     # verde
C_CNN = "#2C6FBB"    # blu
COLORS = {"Euristico": C_HEUR, "NN": C_NN, "CNN": C_CNN}

# Font più grandi: i grafici si vedono più ingranditi nel deck
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def read_metrics(path):
    """Ritorna dict {classe: f1}, accuratezza overall e latenza."""
    f1 = {}
    overall_acc = None
    latency = None
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cls = row["class"]
            if row.get("f1_score"):
                try:
                    f1[cls] = float(row["f1_score"])
                except ValueError:
                    pass
            if cls == "overall":
                overall_acc = float(row["precision"])  # 'overall' mette l'accuracy in precision
                if row.get("avg_latency_ms"):
                    latency = float(row["avg_latency_ms"])
    return f1, overall_acc, latency


def read_pairwise(path):
    d = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.reader(fh):
            if len(r) >= 3 and r[0]:
                d[(r[0], r[1])] = r[2]
    return d


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------
f1_cnn, _, _ = read_metrics(CNN_CSV)
f1_nn, _, _ = read_metrics(NN_CSV)
f1_heur, _, _ = read_metrics(HEUR_CSV)
pw = read_pairwise(PAIRWISE_CSV)


def pct(v): return float(str(v).replace("%", "").strip())
def mb(v): return float(str(v).replace("MB", "").strip())
def ms(v): return float(str(v).replace("ms", "").strip())


acc_cnn = pct(pw[("Scontro: CNN vs NN (Su Mani Rilevate)", "Accuratezza CNN")])
acc_nn = pct(pw[("Scontro: CNN vs NN (Su Mani Rilevate)", "Accuratezza NN")])
acc_heur = pct(pw[("Scontro: CNN vs Euristico (Su Mani Rilevate)", "Accuratezza Euristico")])

size_heur = mb(pw[("Efficienza Spaziale (Memoria)", "Peso Modello Euristico")])
size_nn = mb(pw[("Efficienza Spaziale (Memoria)", "Peso Modello NN")])
size_cnn = mb(pw[("Efficienza Spaziale (Memoria)", "Peso Modello CNN")])
lat_heur = ms(pw[("Efficienza Temporale (Velocità)", "Latenza Media Euristico")])
lat_nn = ms(pw[("Efficienza Temporale (Velocità)", "Latenza Media NN")])
lat_cnn = ms(pw[("Efficienza Temporale (Velocità)", "Latenza Media CNN")])


def mcn(cat, name_a, name_b):
    b = int(pw[(cat, f"Solo {name_a} Corretto (b)")])
    c = int(pw[(cat, f"Solo {name_b} Corretto (c)")])
    p = float(pw[(cat, "McNemar: P-Value")])
    return b, c, p


mcn_data = [
    ("CNN", "NN", mcn("Scontro: CNN vs NN (Su Mani Rilevate)", "CNN", "NN")),
    ("CNN", "Euristico", mcn("Scontro: CNN vs Euristico (Su Mani Rilevate)", "CNN", "Euristico")),
    ("NN", "Euristico", mcn("Scontro: NN vs Euristico (Su Mani Rilevate)", "NN", "Euristico")),
]


# ---------------------------------------------------------------------------
# PLOT 1 - Accuratezza complessiva
# ---------------------------------------------------------------------------
def plot_accuracy():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    names = ["Euristico", "NN", "CNN"]
    vals = [acc_heur, acc_nn, acc_cnn]
    bars = ax.bar(names, vals, color=[COLORS[n] for n in names], width=0.6, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuratezza %")
    ax.set_title("Accuratezza complessiva, confronto leale")
    ax.axhline(100 / 15, ls="--", color="gray", lw=1)
    ax.text(2.45, 100 / 15 + 1, "soglia caso 1/15", color="gray", fontsize=12, ha="right")
    fig.tight_layout()
    p = os.path.join(OUT, "01_accuracy.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def plot_per_class_f1():
    fig, ax = plt.subplots(figsize=(18, 7.5))
    x = np.arange(len(GESTURE_NAMES)); w = 0.27
    ax.bar(x - w, [f1_heur.get(c, 0) for c in GESTURE_NAMES], w, label="Euristico", color=C_HEUR)
    ax.bar(x, [f1_nn.get(c, 0) for c in GESTURE_NAMES], w, label="NN", color=C_NN)
    ax.bar(x + w, [f1_cnn.get(c, 0) for c in GESTURE_NAMES], w, label="CNN", color=C_CNN)
    ax.set_xticks(x); ax.set_xticklabels(GESTURE_NAMES, rotation=45, ha="right")
    ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.05)
    ax.set_title("F1-score per classe, Euristico vs NN vs CNN")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "02_f1.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def plot_mcnemar():
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(mcn_data)); w = 0.35
    b_vals = [d[2][0] for d in mcn_data]
    c_vals = [d[2][1] for d in mcn_data]
    ax.bar(x - w / 2, b_vals, w, label="solo il 1° modello corretto", color="#2C6FBB")
    ax.bar(x + w / 2, c_vals, w, label="solo il 2° modello corretto", color="#E07A5F")
    ymax = max(max(b_vals), max(c_vals))
    for i, (a, bn, (b, c, p)) in enumerate(mcn_data):
        ax.text(x[i] - w / 2, b + ymax * 0.01, str(b), ha="center", fontsize=13, fontweight="bold")
        ax.text(x[i] + w / 2, c + ymax * 0.01, str(c), ha="center", fontsize=13, fontweight="bold")
        ax.text(x[i], max(b, c) + ymax * 0.09, f"p < 1e-10\n→ vince {a if b > c else bn}",
                ha="center", fontsize=13, color="#333", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\nvs\n{bn}" for a, bn, _ in mcn_data])
    ax.set_ylabel("Campioni in disaccordo")
    ax.set_ylim(0, ymax * 1.35)
    ax.set_title("Test di McNemar, disaccordi appaiati e significatività")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p = os.path.join(OUT, "03_mcnemar.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def plot_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    names = ["Euristico", "NN", "CNN"]; cols = [COLORS[n] for n in names]
    lat = [lat_heur, lat_nn, lat_cnn]
    b1 = ax1.bar(names, lat, color=cols, width=0.6)
    for bar, v in zip(b1, lat):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + max(lat) * 0.02, f"{v:.2f} ms",
                 ha="center", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Latenza media in ms")
    ax1.set_title("Analisi temporale, latenza di inferenza")
    ax1.set_ylim(0, max(lat) * 1.25)
    size = [size_heur, size_nn, size_cnn]
    b2 = ax2.bar(names, size, color=cols, width=0.6)
    for bar, v in zip(b2, size):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + max(size) * 0.02, f"{v:.2f} MB",
                 ha="center", fontsize=15, fontweight="bold")
    ax2.set_ylabel("Peso modello su disco in MB")
    ax2.set_title("Analisi spaziale, footprint di memoria")
    ax2.set_ylim(0, max(size) * 1.25)
    fig.tight_layout()
    p = os.path.join(OUT, "04_efficiency.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def plot_tradeoff():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    names = ["Euristico", "NN", "CNN"]
    lat = [lat_heur, lat_nn, lat_cnn]; acc = [acc_heur, acc_nn, acc_cnn]
    size = [size_heur, size_nn, size_cnn]
    areas = [600 + s * 900 for s in size]
    for n, xx, yy, a in zip(names, lat, acc, areas):
        ax.scatter(xx, yy, s=a, color=COLORS[n], alpha=0.75, edgecolor="black", zorder=3)
        ax.annotate(n, (xx, yy), textcoords="offset points", xytext=(12, 10),
                    fontsize=16, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Latenza media in ms, scala logaritmica · più a sinistra = più veloce")
    ax.set_ylabel("Accuratezza %")
    ax.set_title("Trade-off accuratezza vs velocità")
    ax.set_ylim(50, 100)
    ax.text(0.99, 0.02, "NN: miglior trade-off real-time/edge",
            transform=ax.transAxes, ha="right", fontsize=14, color="#3D9970", fontstyle="italic")
    fig.tight_layout()
    p = os.path.join(OUT, "05_tradeoff.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


if __name__ == "__main__":
    for o in [plot_accuracy(), plot_per_class_f1(), plot_mcnemar(),
              plot_efficiency(), plot_tradeoff()]:
        print("  ", o)
