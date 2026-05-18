# cnn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
import os
import copy
import csv
from load_and_plot import *

# --- CONFIGURAZIONE ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_split', 'train')
VAL_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_split', 'val')
TEST_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_split', 'test')

NUM_CLASSES = 15
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 50
PATIENCE = 5

NUM_WORKERS = min(8, os.cpu_count())

GESTURE_NAMES = [
    'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
    'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP',
    'THREE', 'TWO', 'NO GESTURE'
]


# ---------------------------------------------------------------------------
# 1. MODELLO MOBILENET V3 (FULL FINE-TUNING)
# ---------------------------------------------------------------------------
class HandGestureFineTune(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = mobilenet_v3_small(weights=weights)

        # RIMOSSO IL CONGELAMENTO DEI PESI:
        # Ora l'intero backbone può aggiornarsi per riconoscere le mani.

        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# 2. CLASSE GESTORE: TRAINING & TEST OTTIMIZZATI
# ---------------------------------------------------------------------------
class GestureClassifierCNN:
    def __init__(self, num_classes=15, img_size=224, learning_rate=1e-4):  # LR Abbassato a 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.model = HandGestureFineTune(num_classes).to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Ottimizzatore: aggiornato per passare tutti i parametri e gestire il weight_decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Scheduler: Dimezza il LR (factor=0.5) se la Val Loss non migliora per 3 epoche (patience=3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

        # Trasformazioni: Aggiunte rotazioni e traslazioni specifiche per l'orientamento delle mani
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train(self, train_dir, val_dir, epochs=50, batch_size=128, patience=10):
        print(f"\n--- Inizio Training Ottimizzato su {self.device} ---")

        train_data = datasets.ImageFolder(train_dir, transform=self.train_transform)
        val_data = datasets.ImageFolder(val_dir, transform=self.eval_transform)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                  pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True)

        best_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        counter = 0

        for epoch in range(epochs):
            self.model.train()
            running_loss, correct = 0.0, 0

            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

            self.model.eval()
            v_loss, v_correct = 0.0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                        outputs = self.model(images)
                        v_loss += self.criterion(outputs, labels).item() * images.size(0)
                        v_correct += (outputs.argmax(1) == labels).sum().item()

            ep_val_loss = v_loss / len(val_data)
            self.history['loss'].append(running_loss / len(train_data))
            self.history['accuracy'].append(correct / len(train_data))
            self.history['val_loss'].append(ep_val_loss)
            self.history['val_accuracy'].append(v_correct / len(val_data))

            print(
                f"Epoch {epoch + 1:2d}/{epochs} | Val Acc: {self.history['val_accuracy'][-1]:.3f} | Val Loss: {ep_val_loss:.4f}")

            # STEP DELLO SCHEDULER: analizza la loss di validazione corrente
            self.scheduler.step(ep_val_loss)

            if ep_val_loss < best_loss:
                best_loss = ep_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"\nEarly stopping innescato all'epoca {epoch + 1}.")
                    break

        self.model.load_state_dict(best_model_wts)
        return self.history

    def evaluate(self, test_dir, batch_size=64,
                 output_csv="test_results_cnn2d.csv",
                 metrics_csv="metrics_summary_cnn.csv"):
        print(f"\n--- Valutazione su Test Set (Batch Size: {batch_size}) ---")
        test_data = datasets.ImageFolder(test_dir, transform=self.eval_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

        self.model.eval()
        results = []
        all_preds = []
        all_labels = []
        global_idx = 0
        all_batch_latencies = []  # latenze per batch, poi media ponderata

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device, non_blocking=True)
                bs = images.size(0)

                t0 = time.perf_counter()
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                img_latency_ms = (time.perf_counter() - t0) * 1000 / bs

                all_batch_latencies.extend([img_latency_ms] * bs)  # una voce per immagine

                preds = outputs.argmax(1).cpu().numpy()
                labels_cpu = labels.numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_cpu)

                for j in range(bs):
                    results.append({
                        'sample_index': global_idx,
                        'true_label': labels_cpu[j],
                        'pred_label': preds[j],
                        'is_correct': 1 if preds[j] == labels_cpu[j] else 0,
                        'latency_ms': img_latency_ms
                    })
                    global_idx += 1

        # ── CSV per immagine (McNemar) ────────────────────────────────────────
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[OK] {len(results)} risultati per-immagine salvati in: {output_csv}")

        # ── Latenza media globale ─────────────────────────────────────────────
        avg_latency = np.mean(all_batch_latencies)
        print(f"Latenza media per immagine: {avg_latency:.3f} ms")

        # ── CSV metriche aggregate ────────────────────────────────────────────
        save_metrics_csv(all_labels, all_preds, GESTURE_NAMES,
                         avg_latency_ms=avg_latency,
                         output_csv=metrics_csv)

        report = classification_report(all_labels, all_preds, target_names=GESTURE_NAMES)
        print("\nReport Finale:\n", report)

        return confusion_matrix(all_labels, all_preds)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    classifier = GestureClassifierCNN(num_classes=NUM_CLASSES)

    history_raw = classifier.train(
        train_dir=TRAIN_DIR, val_dir=VAL_DIR,
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE
    )

    class MockHistory:
        def __init__(self, h): self.history = h

    # Salva i grafici su disco invece di mostrarli interattivamente
    plot_training_history(MockHistory(history_raw), save_path="plots/cnn_training_history.png")

    cm = classifier.evaluate(
        TEST_DIR,
        batch_size=64,
        output_csv="results_cnn_2d.csv",
        metrics_csv="metrics_summary_cnn.csv"
    )

    plot_confusion_matrix(cm, GESTURE_NAMES, save_path="plots/cnn_confusion_matrix.png")

    torch.save(classifier.model.state_dict(), "gesture_cnn_finetuned.pth")
    print("\n[FINISH] Pipeline completata.")