"""
Diagramma della pipeline del progetto (immagine singola).
Output: utils/presentation_plots/00_pipeline.png

Run: python utils/presentation_plots/make_pipeline.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "00_pipeline.png")

# Palette coerente col resto della presentazione
DARK = "#1A2A3A"
BLUE = "#2C6FBB"
GREEN = "#3D9970"
TERRA = "#E07A5F"
GRAY = "#6B7785"
LIGHT = "#EEF3FA"
DATA = "#F4C95D"

fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")


def box(x, y, w, h, text, fc, tc="white", fs=12, bold=True, ec=None, rounded=0.08):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={rounded}",
                       linewidth=1.6, edgecolor=ec or fc, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            zorder=3, wrap=True)
    return (x, y, w, h)


def arrow(p1, p2, color=GRAY, style="-|>", lw=2.2, conn="arc3,rad=0"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                        color=color, lw=lw, zorder=1,
                        connectionstyle=conn)
    ax.add_patch(a)


def right(b):  # punto centro-destro
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


# Titolo
ax.text(50, 54, "Pipeline del progetto — Hand Gesture Recognition",
        ha="center", fontsize=19, color=DARK, fontweight="bold")

# ---- Colonna 1: dato grezzo ----
b_raw = box(2, 36, 17, 8, "Dataset HaGRID\n(immagini 512px)\n15 gesti", DATA, tc=DARK, fs=12)

# ---- Split ----
b_split = box(2, 24, 17, 7, "Split 80/10/10\ntrain / val / test\n(splitfolders)", LIGHT, tc=DARK, fs=11, ec=GRAY)
arrow(bottom(b_raw), top(b_split))

# ---- MediaPipe feature extraction ----
b_mp = box(26, 34, 19, 10,
           "MediaPipe Hands\nestrazione landmark\n→ 73 feature/mano\n(63 coord + 10 angoli)",
           BLUE, fs=11)
arrow(right(b_raw), left(b_mp))

# ---- CSV keypoint ----
b_csv = box(26, 22, 19, 7, "CSV keypoint\ntrain/val/test_keypoint.csv", LIGHT, tc=DARK, fs=11, ec=GRAY)
arrow(bottom(b_mp), top(b_csv))

# ---- Tre rami classificatori ----
# Euristico (da CSV)
b_heur = box(52, 44, 22, 8.5,
             "EURISTICO\nregole geometriche + omografia\n0 parametri", TERRA, fs=11)
arrow(right(b_csv), left(b_heur), color=TERRA, conn="arc3,rad=-0.25")

# NN (da CSV)
b_nn = box(52, 30.5, 22, 8.5,
           "NN su keypoint\nFC 73-50-50-50-15\n9.565 parametri", GREEN, fs=11)
arrow(right(b_csv), left(b_nn), color=GREEN, conn="arc3,rad=0.0")

# CNN (da immagini, ramo diretto)
b_cnn = box(52, 17, 22, 8.5,
            "CNN end-to-end\nMobileNetV3-Small (224x224)\n~1,52 M parametri", BLUE, fs=11)
# freccia diretta dalle immagini (bypassa MediaPipe)
arrow((19, 39), (52, 21.2), color=BLUE, conn="arc3,rad=-0.32", lw=2.4)
ax.text(33, 13.5, "ramo CNN: pixel grezzi\n(indipendente da MediaPipe)",
        ha="center", fontsize=9.5, color=BLUE, fontstyle="italic")

# ---- Valutazione ----
b_eval = box(80, 30.5, 18, 8.5,
             "Valutazione\nprecision / recall / F1\nlatenza · memoria", DARK, fs=11)
arrow(right(b_heur), (80, 34.7), color=GRAY, conn="arc3,rad=0.22")
arrow(right(b_nn), left(b_eval), color=GRAY)
arrow(right(b_cnn), (80, 34.7), color=GRAY, conn="arc3,rad=-0.22")

# ---- Analisi statistica ----
b_stat = box(80, 17, 18, 8.5,
             "Analisi statistica\nMcNemar (pairwise)\ntrade-off acc/velocità", LIGHT, tc=DARK, fs=11, ec=GRAY)
arrow(bottom(b_eval), top(b_stat))

# ---- App live ----
b_app = box(80, 44, 18, 7,
            "App live (webcam)\nEuristico / NN\nreal-time", GREEN, fs=11)
arrow(top(b_heur), (89, 44), color=GREEN, conn="arc3,rad=0.15")
arrow(top(b_nn), (87, 44), color=GREEN, conn="arc3,rad=-0.15")

# Legenda fasi
ax.text(2, 7.5, "Fasi:", fontsize=11, color=DARK, fontweight="bold")
for i, (c, t) in enumerate([
    (DATA, "dati"), (BLUE, "estrazione / modello CNN"),
    (TERRA, "Euristico"), (GREEN, "NN / app"), (DARK, "valutazione")]):
    ax.add_patch(FancyBboxPatch((10 + i * 17, 6.6), 1.6, 1.6,
                 boxstyle="round,pad=0.02", facecolor=c, edgecolor=c))
    ax.text(12 + i * 17, 7.4, t, fontsize=9.5, va="center", color=DARK)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK]", OUT)
