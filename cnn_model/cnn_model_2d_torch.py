import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
import os
from nn_model.load_and_plot import *

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cartelle generate dallo split delle immagini
TRAIN_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_splitted', 'train')
VAL_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_splitted', 'val')
TEST_DIR = os.path.join(project_root, 'dataset', 'hagrid_dataset_splitted', 'test')

# --- PARAMETRI ---
NUM_CLASSES = 14
IMG_SIZE = 128  # Dimensione a cui vengono ridimensionate le immagini
BATCH_SIZE = 32
EPOCHS = 50

# Nomi delle gesture nell'ordine esatto delle cartelle (0, 1, 2...)
NOMI_GESTURE = [
    'CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK',
    'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP', 'THREE', 'TWO'
]


class HandGestureCNN(nn.Module):
    """
    Architettura CNN custom per il riconoscimento delle gesture.
    Progettata per bilanciare accuratezza e costo computazionale.
    """

    def __init__(self, num_classes):
        super(HandGestureCNN, self).__init__()
        # Estrattore di feature (Convoluzioni)
        self.features = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 32 x 64 x 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: 128 x 16 x 16
        )

        # Classificatore (Livelli densi)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Per prevenire l'overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class GestureClassifierCNN:
    def __init__(self, num_classes=14, img_size=128, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.model = HandGestureCNN(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

        # Trasformazioni per le immagini
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train(self, train_dir, val_dir, epochs=50, batch_size=32, patience=7):
        """
        Addestramento usando ImageFolder per caricare direttamente dalle cartelle splittate.
        """
        print(f"--- Training CNN su {self.device} ---")

        train_data = datasets.ImageFolder(train_dir, transform=self.transform)
        val_data = datasets.ImageFolder(val_dir, transform=self.transform)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        best_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss, correct = 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()

            # Validazione
            self.model.eval()
            v_loss, v_correct = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    v_loss += loss.item() * images.size(0)
                    _, pred = torch.max(outputs, 1)
                    v_correct += (pred == labels).sum().item()

            metrics = {
                'loss': train_loss / len(train_data),
                'accuracy': correct / len(train_data),
                'val_loss': v_loss / len(val_data),
                'val_accuracy': v_correct / len(val_data)
            }

            for k, v in metrics.items(): self.history[k].append(v)
            print(f"Epoca {epoch + 1}: Loss {metrics['loss']:.4f} | Val Acc {metrics['val_accuracy']:.4f}")

            # Early Stopping
            if metrics['val_loss'] < best_loss:
                best_loss = metrics['val_loss']
                counter = 0
                torch.save(self.model.state_dict(), "best_cnn_model.pth")
            else:
                counter += 1
                if counter >= patience: break

        return self.history

    def evaluate(self, test_dir):
        """Valutazione finale e calcolo tempi di inferenza."""
        test_data = datasets.ImageFolder(test_dir, transform=self.transform)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        self.model.eval()

        all_preds, all_labels = [], []
        latencies = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)

                start = time.time()
                outputs = self.model(images)
                latencies.append(time.time() - start)

                _, pred = torch.max(outputs, 1)
                all_preds.append(pred.item())
                all_labels.append(labels.item())

        avg_lat = np.mean(latencies) * 1000
        print(f"Latenza media CNN: {avg_lat:.2f} ms")
        print(classification_report(all_labels, all_preds))
        return confusion_matrix(all_labels, all_preds)

if __name__ == "__main__":
    print("=" * 50)
    print(" PIPELINE ADDESTRAMENTO CNN (RGB IMAGES) ")
    print("=" * 50)

    # 1. Inizializzazione del classificatore
    # La CNN opererà su immagini RGB grezze, diversamente dal sistema a keypoint[cite: 33].
    classifier = GestureClassifierCNN(num_classes=NUM_CLASSES, img_size=IMG_SIZE)

    # Verifica disponibilità GPU (fondamentale per la sostenibilità della CNN)
    print(f"Dispositivo utilizzato: {classifier.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Addestramento con Early Stopping
    # Questo processo richiederà più tempo rispetto alla NN per l'elevato numero di parametri.
    history_raw = classifier.train(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        patience=7
    )

    # Convertiamo la history per renderla compatibile con le tue funzioni di plot
    class MockHistory:
        def __init__(self, h): self.history = h

    history_obj = MockHistory(history_raw)
    plot_training_history(history_obj)

    # 3. Valutazione Finale sul Test Set
    # Qui misureremo la latenza media (ms) per il confronto on-device[cite: 22, 59].
    print("\nAvvio valutazione finale sul Test Set...")
    cm = classifier.evaluate(TEST_DIR)

    # 4. Visualizzazione Risultati
    plot_confusion_matrix(cm, NOMI_GESTURE)

    # 5. Salvataggio finale del modello
    final_model_path = "gesture_cnn_final.pth"
    torch.save(classifier.model.state_dict(), final_model_path)
    print(f"\n[OK] Modello CNN salvato in: {final_model_path}")
    print("=" * 50)