import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import time
from load_and_plot import *
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(project_root, "dataset", "train_keypoint.csv")
VAL_FILE = os.path.join(project_root, "dataset", "val_keypoint.csv")
TEST_FILE = os.path.join(project_root, "dataset", "test_keypoint.csv")

NOMI_GESTURE = ['CALL', 'DISLIKE', 'FIST', 'FOUR', 'LIKE', 'MUTE', 'OK', 'POINTING UP', 'OPEN PALM', 'VICTORY', 'ROCK', 'STOP', 'THREE', 'TWO', 'NO GESTURE']

class GestureClassifierNN:
    def __init__(self, input_dim=42, num_classes=14, learning_rate=0.001):
        """
        Inizializza la classe, definisce le dimensioni e costruisce il modello.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        """
        Costruisce e compila l'architettura della rete neurale (3 layer da 50 neuroni).
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=(self.input_dim,)),
            layers.Dense(50, activation='relu', name='dense_1'),
            layers.Dense(50, activation='relu', name='dense_2'),
            layers.Dense(50, activation='relu', name='dense_3'),
            layers.Dense(self.num_classes, activation='softmax', name='output_gestures')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Esegue l'addestramento del modello e salva i tempi di esecuzione.
        """
        print(f"--- Inizio Addestramento per {epochs} epoche ---")
        start_time = time.time()

        # model.fit gestisce automaticamente i cicli di training
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1  # Imposta a 0 se non vuoi stampare i log per ogni epoca
        )

        end_time = time.time()
        training_time = end_time - start_time
        print(f"--- Addestramento completato in {training_time:.2f} secondi ---")

        return self.history

    def evaluate_model(self, X_test, y_test):
        """
        Valuta il modello sul test set, estraendo metriche statistiche e tempi di inferenza.
        Questi sono i dati che ti serviranno per i test statistici e le conclusioni.
        """
        print("--- Avvio Valutazione ---")

        # 1. Misurazione del tempo di Inferenza (Cruciale per il real-time on-device)
        start_inf_time = time.time()
        y_pred_probs = self.model.predict(X_test, verbose=0)
        end_inf_time = time.time()

        # Calcolo del tempo medio per SINGOLA immagine in millisecondi
        total_inf_time_ms = (end_inf_time - start_inf_time) * 1000
        avg_inf_time_ms = total_inf_time_ms / len(X_test)

        # 2. Convertire le probabilità nella classe predetta (quella con probabilità maggiore)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # 3. Metriche Statistiche (Recall, Precision, F1-Score)
        # Il parametro zero_division=0 evita avvisi se una classe non viene mai predetta
        report = classification_report(y_test, y_pred, zero_division=0)

        # 4. Matrice di Confusione
        cm = confusion_matrix(y_test, y_pred)

        print(f"Tempo di inferenza medio: {avg_inf_time_ms:.4f} ms per frame")
        print("\nReport di Classificazione (Include Recall per ogni classe):\n", report)

        # Restituiamo un dizionario con tutti i dati per poterli salvare o plottare
        return {
            "average_inference_time_ms": avg_inf_time_ms,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": y_pred  # Utili se vorrai fare il test di McNemar in seguito
        }

    def summary(self):
        """Stampa la struttura della rete e il conteggio dei parametri."""
        self.model.summary()


if __name__ == "__main__":
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print(f"Numero di GPU rilevate: {num_gpus}")
    if num_gpus == 0:
        exit(-1)

    # 2. Mostra i dettagli di tutti i dispositivi (CPU e GPU)
    print("\nDettagli dei dispositivi di calcolo:")
    for device in tf.config.list_physical_devices():
        print(device)

    X_train, y_train = load_dataset(TRAIN_FILE, label_column_index=0)
    X_val, y_val = load_dataset(VAL_FILE, label_column_index=0)
    X_test, y_test = load_dataset(TEST_FILE, label_column_index=0)

    classifier = GestureClassifierNN()

    # Mostra l'architettura per confermare che sia leggera per il mobile
    classifier.summary()

    # --- 4. ADDESTRAMENTO ---
    # Usa un batch size piccolo (es. 32 o 64) e un buon numero di epoche
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=80,
        batch_size=32
    )

    plot_training_history(history)

    # --- 5. VALUTAZIONE E TEST STATISTICI ---
    # Questa è la parte cruciale per il tuo paper: estrarre le metriche sul test set
    risultati = classifier.evaluate_model(X_test, y_test)

    # Visualizza la matrice di confusione
    plot_confusion_matrix(risultati["confusion_matrix"], NOMI_GESTURE)

    # --- 6. SALVATAGGIO MODELLO (Opzionale ma consigliato) ---
    # Salva il modello in formato Keras per conversioni future in TFLite
    salvataggio_path = "gesture_nn_model.keras"
    classifier.model.save(salvataggio_path)
    print(f"\nModello salvato con successo in: {salvataggio_path}")
    print("=== PIPELINE COMPLETATA ===")