import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time
from nn_model.load_and_plot import *
import copy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cartelle generate dallo split delle immagini
TRAIN_FILE = os.path.join(project_root, "dataset", "train_keypoint.csv")
VAL_FILE = os.path.join(project_root, "dataset", "val_keypoint.csv")
TEST_FILE = os.path.join(project_root, "dataset", "test_keypoint.csv")

NOMI_GESTURE = ['CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK', 'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK',
                'STOP', 'THREE', 'TWO']

NUM_CLASSES = 14
IMG_SIZE = 128  # Dimensione a cui vengono ridimensionate le immagini
BATCH_SIZE = 32
EPOCHS = 50


class GestureCNN1D(nn.Module):
    def __init__(self, input_dim=63, num_classes=14):
        super(GestureCNN1D, self).__init__()

        # Una 1D-CNN si aspetta un input (Batch, Canali, Lunghezza)
        # Nel nostro caso: Canali = 1, Lunghezza = 63 (o 42)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Dimezza la lunghezza

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Dimezza ancora
        )

        # Calcoliamo la dimensione di uscita dopo le convoluzioni
        # Se input_dim=63 -> dopo MaxPool(2) diventa 31 -> dopo MaxPool(2) diventa 15
        # Se input_dim=42 -> diventa 21 -> poi 10
        reduced_dim = input_dim // 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x arriva come (Batch, 63) -> dobbiamo portarlo a (Batch, 1, 63)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x


class GestureClassifierCNN1D:
    def __init__(self, input_dim=63, num_classes=14, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GestureCNN1D(input_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=10):
        """
        Addestramento con Early Stopping: ferma il training se la val_loss non migliora.
        """
        print(f"--- Training 1D-CNN su {self.device} ---")

        # Conversione dati in Tensori
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

        # Variabili per Early Stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())  # Salvataggio iniziale dei pesi

        start_time = time.time()

        for epoch in range(epochs):
            # --- FASE DI TRAINING ---
            self.model.train()
            t_loss, t_correct = 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                t_loss += loss.item() * inputs.size(0)
                t_correct += (torch.argmax(outputs, 1) == labels).sum().item()

            # --- FASE DI VALIDAZIONE ---
            self.model.eval()
            v_loss, v_correct = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    v_loss += loss.item() * inputs.size(0)
                    v_correct += (torch.argmax(outputs, 1) == labels).sum().item()

            # Calcolo medie dell'epoca
            epoch_train_loss = t_loss / len(X_train)
            epoch_val_loss = v_loss / len(X_val)
            epoch_val_acc = v_correct / len(X_val)

            self.history['loss'].append(epoch_train_loss)
            self.history['accuracy'].append(t_correct / len(X_train))
            self.history['val_loss'].append(epoch_val_loss)
            self.history['val_accuracy'].append(epoch_val_acc)

            print(
                f"Epoca {epoch + 1}/{epochs} | Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

            # --- LOGICA EARLY STOPPING ---
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                # Se miglioriamo, salviamo questi pesi come "i migliori finora"
                best_model_wts = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n[!] Early Stopping: la val_loss non migliora da {patience} epoche. Stop.")
                    break

        # Ripristiniamo i pesi dell'epoca migliore prima di finire
        self.model.load_state_dict(best_model_wts)
        print(f"--- Training completato. Pesi ripristinati alla migliore Val Loss ({best_val_loss:.4f}) ---")

        return self.history

    def evaluate(self, X_test, y_test):
        """
        Valuta la 1D-CNN sul test set.
        Calcola l'accuratezza, la matrice di confusione e il tempo di inferenza medio.
        """
        print("\n--- Avvio Valutazione 1D-CNN ---")

        # 1. Preparazione dati
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()

        all_preds = []
        latencies = []

        # 2. Inferenza e misurazione tempi
        # Eseguiamo le predizioni una ad una per simulare il comportamento "live"
        # e avere un tempo di inferenza reale per singolo frame.
        with torch.no_grad():
            for i in range(X_test_t.size(0)):
                # Prendi un singolo campione e aggiungi la dimensione del batch (1, 63)
                sample = X_test_t[i].unsqueeze(0)

                start_time = time.time()
                output = self.model(sample)
                end_time = time.time()

                latencies.append(end_time - start_time)

                # Prendi la classe predetta
                _, pred = torch.max(output, 1)
                all_preds.append(pred.item())

        # 3. Calcolo metriche
        avg_inference_time_ms = np.mean(latencies) * 1000
        y_pred = np.array(all_preds)

        # Generazione report e matrice
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Tempo di inferenza medio (1D-CNN): {avg_inference_time_ms:.4f} ms per frame")
        print("\nReport di Classificazione:\n", report)

        return {
            "average_inference_time_ms": avg_inference_time_ms,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": y_pred
        }


if __name__ == "__main__":
    print("=" * 50)
    print(" PIPELINE ADDESTRAMENTO CNN (RGB IMAGES) ")
    print("=" * 50)

    # 1. Inizializzazione del classificatore
    # La CNN opererà su immagini RGB grezze, diversamente dal sistema a keypoint[cite: 33].
    classifier = GestureClassifierCNN1D(num_classes=NUM_CLASSES)

    # Verifica disponibilità GPU (fondamentale per la sostenibilità della CNN)
    print(f"Dispositivo utilizzato: {classifier.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Addestramento con Early Stopping
    # Questo processo richiederà più tempo rispetto alla NN per l'elevato numero di parametri.
    X_train, y_train = load_dataset(TRAIN_FILE, label_column_index=0)
    X_val, y_val = load_dataset(VAL_FILE, label_column_index=0)
    X_test, y_test = load_dataset(TEST_FILE, label_column_index=0)

    history_raw = classifier.train(
        X_train = X_train,
        y_train = y_train,
        X_val = X_val,
        y_val = y_val,
    )

    # Convertiamo la history per renderla compatibile con le tue funzioni di plot
    class MockHistory:
        def __init__(self, h): self.history = h

    history_obj = MockHistory(history_raw)
    plot_training_history(history_obj)

    # 3. Valutazione Finale sul Test Set
    # Qui misureremo la latenza media (ms) per il confronto on-device[cite: 22, 59].
    print("\nAvvio valutazione finale sul Test Set...")
    cm = classifier.evaluate(
        X_test=X_test,
        y_test=y_test
    )

    # 4. Visualizzazione Risultati
    plot_confusion_matrix(cm, NOMI_GESTURE)

    # 5. Salvataggio finale del modello
    final_model_path = "gesture_cnn_1d_final.pth"
    torch.save(classifier.model.state_dict(), final_model_path)
    print(f"\n[OK] Modello CNN salvato in: {final_model_path}")
    print("=" * 50)