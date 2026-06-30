# LiveHandRecognition
This is my project for Computer Vision exam @ Federico II. 
Real-time hand-gesture recognition from a webcam, comparing **three different
approaches** to the same 15-class problem: a hand-crafted **heuristic**
classifier, a small **neural network (NN)** over hand keypoints, and a
**convolutional neural network (CNN)** fine-tuned end-to-end on raw images.

The project trains, evaluates, and statistically compares the three models, and
ships a live demo app that runs any of them on the webcam feed.

## Gestures

15 classes:

```
CALL, DISLIKE, FIST, FOUR, LIKE, MUTE, OK, POINTING UP,
OPEN PALM, VICTORY, ROCK, STOP, THREE, TWO, NO GESTURE
```

## The three models

| Model | Input | How it works | Weights file |
|-------|-------|--------------|--------------|
| **Heuristic** | MediaPipe hand keypoints | Hand-written if/else rules over finger states (straight / bent / crossed), plus a homography correction for the thumb | none (code only) |
| **NN** | 73 features (63 normalized keypoint coords + 10 finger angles) | 3 hidden layers of 50 ReLU units | `nn_model/gesture_nn_model.pth` |
| **CNN** | Raw 224×224 RGB frame | MobileNetV3-Small fine-tuned (no MediaPipe needed) | `cnn_model/gesture_cnn_finetuned.pth` |

The heuristic and NN models rely on [MediaPipe Hands](https://developers.google.com/mediapipe)
to extract 21 hand landmarks; the CNN works directly on the image.

## Project layout

```
app.py                          Live webcam demo (pick the model at the top of __main__)
requirements.txt

heuristic_model/
    heuristic_model.py          Rule-based classifier + evaluation pipeline
    HomographyRef.png           Reference points for the thumb homography correction

nn_model/
    nn_model.py                 Keypoint NN: training, evaluation, inference
    load_and_plot.py            Data loading + plotting/metrics helpers

cnn_model/
    cnn_model.py                MobileNetV3 fine-tuning: training, evaluation
    load_and_plot.py            Data loading + plotting/metrics helpers

utils/
    utility.py                  Shared landmark math + drawing helpers
    dataset_generator.py        Build the keypoint CSVs from the image dataset
    test_head_to_head.py        Run all 3 models on the test set -> head_to_head_results.csv
    mcnemar_test.py             Pairwise McNemar tests + size/latency benchmark

assets/                         Figures, confusion matrices, demo GIFs
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
# PyTorch is not pinned in requirements.txt: install the build matching your
# CUDA version from https://pytorch.org (CPU-only also works for inference)
pip install torch torchvision
```

The training and statistical scripts also need `statsmodels`
(`pip install statsmodels`).

## Dataset

The models are trained on a 15-class subset derived from the
[HaGRID](https://github.com/hukenovs/hagrid) hand-gesture dataset.

Expected layout (the `dataset/` folder is git-ignored):

```
dataset/
    hagrid_dataset_512/         Source images, one folder per class "{label}_{gesture}"
    hagrid_dataset_split/       train / val / test split (created by the generator)
    train_keypoint.csv          Keypoint features for the heuristic and NN models
    val_keypoint.csv
    test_keypoint.csv
```

Generate the split and the keypoint CSVs:

```bash
cd utils
python dataset_generator.py
```

This splits the images 80/10/10 and runs MediaPipe over every image to produce
the keypoint CSVs (label + 63 coordinates + 10 angles per row).

## Running the live demo

Edit the model selector at the top of `__main__` in `app.py`:

```python
model_type = HEURISTIC   # or NN, or CNN
```

then run:

```bash
python app.py
```

Controls:

- **q** — quit
- **k** — enable keypoint logging mode
- **n** — disable logging mode
- **0–9** — set the class number while logging

The window shows the detected hand, the predicted gesture, the live FPS, and the
name of the active model.

## Training

```bash
# Keypoint NN (requires a CUDA GPU)
python -m nn_model.nn_model

# CNN (MobileNetV3 fine-tuning)
python -m cnn_model.cnn_model

# Heuristic model has no training; evaluate it directly
python -m heuristic_model.heuristic_model
```

Each script writes a per-sample CSV (for McNemar), an aggregate metrics CSV
(precision / recall / f1 + average latency), and saves plots under `plots/`.

## Model comparison

```bash
# 1. Run all three models on the test set
python -m utils.test_head_to_head      # -> head_to_head_results.csv

# 2. Pairwise McNemar tests + on-disk size + inference latency
python -m utils.mcnemar_test           # -> pairwise_results_summary.csv
```

The McNemar test reports whether the accuracy gap between each pair of models is
statistically significant (p < 0.05), and benchmarks each model's size on disk
and per-frame inference latency.
