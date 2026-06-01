import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
import csv

from nn_model.load_and_plot import *
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(project_root, "dataset", "train_keypoint.csv")
VAL_FILE = os.path.join(project_root, "dataset", "val_keypoint.csv")
TEST_FILE = os.path.join(project_root, "dataset", "test_keypoint.csv")

GESTURE_NAMES = [
    'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
    'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP', 'THREE', 'TWO', 'NO GESTURE'
]

INPUT_DIM = 73
NUM_CLASSES = 15

class GestureNN(nn.Module):
    """Definizione vera e propria del modello in PyTorch"""

    def __init__(self, input_dim, num_classes):
        super(GestureNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class GestureClassifierNN:
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES, learning_rate=0.001):
        """
        Inizializza la classe e sposta il modello sulla GPU (se disponibile).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GestureNN(input_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history_dict = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=10):
        """
        Esegue l'addestramento con EARLY STOPPING e salvataggio dei pesi migliori.
        """
        print(f"--- Inizio Addestramento per {epochs} epoche su {self.device} ---")

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

        start_time = time.time()

        # Variabili per l'Early Stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(epochs):
            # --- FASE DI TRAINING ---
            self.model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            # --- FASE DI VALIDATION ---
            self.model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            # Calcolo delle metriche
            epoch_loss = running_loss / total_train
            epoch_acc = correct_train / total_train
            epoch_val_loss = val_loss / total_val
            epoch_val_acc = correct_val / total_val

            # Salvataggio storia
            self.history_dict['loss'].append(epoch_loss)
            self.history_dict['accuracy'].append(epoch_acc)
            self.history_dict['val_loss'].append(epoch_val_loss)
            self.history_dict['val_accuracy'].append(epoch_val_acc)

            print(
                f"Epoca [{epoch + 1}/{epochs}] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

            # --- EARLY STOPPING ---
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                # Salva una copia profonda dei pesi attuali
                best_model_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"\n[!] Early Stopping innescato all'epoca {epoch + 1}.")
                print(f"[!] Nessun miglioramento della validation loss per {patience} epoche consecutive.")
                break  # Interrompe il ciclo delle epoche

        end_time = time.time()
        print(f"--- Addestramento completato in {end_time - start_time:.2f} secondi ---")

        # RIPRISTINO DEI PESI MIGLIORI
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            print("-> Pesi del modello ripristinati all'epoca con la migliore Validation Loss.")

        class MockHistory:
            pass

        history_obj = MockHistory()
        history_obj.history = self.history_dict
        return history_obj

    def evaluate_model(self, X_test, y_test,
                       output_csv="test_results_nn.csv",
                       metrics_csv="metrics_summary_nn.csv"):
        """
        Valuta il modello sul test set.
        Produce:
        - CSV per-immagine con is_correct e latency_ms (per McNemar)
        - CSV metriche aggregate (precision/recall/f1 + latenza media)
        """
        print("--- Avvio Valutazione ---")

        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()

        # ── Inferenza per-sample per latenza realistica ───────────────────────
        # La NN su landmark è così piccola che un batch unico non è rappresentativo
        # della latenza reale nel loop webcam — misuriamo sample per sample
        latencies = []
        all_outputs = []

        with torch.no_grad():
            for i in range(len(X_test_t)):
                sample = X_test_t[i].unsqueeze(0)  # shape (1, input_dim)
                t0 = time.perf_counter()
                out = self.model(sample)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
                all_outputs.append(out)

        outputs = torch.cat(all_outputs, dim=0)
        y_pred = outputs.argmax(1).cpu().numpy()
        avg_lat = float(np.mean(latencies))

        print(f"Latenza media per frame: {avg_lat:.4f} ms")

        # ── CSV per-immagine (McNemar) ────────────────────────────────────────
        results = [
            {
                'sample_index': i,
                'true_label': int(y_test[i]),
                'pred_label': int(y_pred[i]),
                'is_correct': 1 if y_pred[i] == y_test[i] else 0,
                'latency_ms': round(latencies[i], 4)
            }
            for i in range(len(y_test))
        ]
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[OK] {len(results)} risultati per-immagine salvati in: {output_csv}")

        # ── CSV metriche aggregate ────────────────────────────────────────────
        save_metrics_csv(
            y_test, y_pred,
            class_names=GESTURE_NAMES,
            avg_latency_ms=avg_lat,
            output_csv=metrics_csv
        )

        report = classification_report(y_test, y_pred,
                                       target_names=GESTURE_NAMES, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        print("\nReport di Classificazione:\n", report)

        return {
            "average_inference_time_ms": avg_lat,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": y_pred
        }

    def summary(self):
        """Stampa la struttura della rete e il conteggio dei parametri."""
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotale parametri addestrabili: {total_params}")

    def save(self, path):
        """Salva i pesi del modello in formato PyTorch."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Carica i pesi del modello salvato dal file .pth"""
        # map_location serve nel caso in cui addestri su GPU ma testi su un PC senza GPU
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()  # Mette il modello in modalità inferenza (disabilita dropout/batchnorm)
        print(f"Pesi del modello caricati da: {path}")

    def predict(self, single_landmark_list):
        """
        Prende in input la lista di 42 keypoint normalizzati,
        la converte in tensore e restituisce l'ID della classe predetta.
        """
        self.model.eval()

        # 1. PyTorch si aspetta i dati a "lotti" (batch).
        # Convertiamo la lista in un tensore 2D di forma (1, 42) aggiungendo le parentesi quadre
        input_tensor = torch.tensor([single_landmark_list], dtype=torch.float32).to(self.device)

        # 2. Forward pass senza calcolare i gradienti (molto più veloce)
        with torch.no_grad():
            outputs = self.model(input_tensor)

            # 3. Prendi l'indice con la probabilità massima
            _, predicted = torch.max(outputs.data, 1)

        # 4. Converti il tensore in un normale intero Python
        return predicted.item()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Errore: Nessuna GPU CUDA rilevata.")
        exit(-1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    X_train, y_train = load_dataset(TRAIN_FILE, label_column_index=0)
    X_val, y_val = load_dataset(VAL_FILE, label_column_index=0)
    X_test, y_test = load_dataset(TEST_FILE, label_column_index=0)

    classifier = GestureClassifierNN()
    classifier.summary()

    history = classifier.train(
        X_train, y_train,
        X_val,   y_val,
        epochs=80, batch_size=32
    )

    plot_training_history(history, save_path="plots/nn_training_history.png")

    risultati = classifier.evaluate_model(
        X_test, y_test,
        output_csv="test_results_nn.csv",
        metrics_csv="metrics_summary_nn.csv"
    )

    plot_confusion_matrix(risultati["confusion_matrix"], GESTURE_NAMES,
                          save_path="plots/nn_confusion_matrix.png")

    classifier.save("gesture_nn_model.pth")
    print("\nModello salvato. === PIPELINE COMPLETATA ===")
