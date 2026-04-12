import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import os
import numpy as np

# --- COSTANTI DEI FILE ---



# --- 1. FUNZIONE DI LETTURA DATI ---
def load_dataset(file_path, label_column_index=0, shuffle_data=True):
    """
    Legge un file CSV e separa le features (X) dalle etichette (y).

    Parametri:
    - file_path: Il percorso del file CSV.
    - label_column_index: L'indice della colonna che contiene la classe (default: 0, la prima colonna).
    """

    print(f"Caricamento dati da: {file_path}...")
    df = pd.read_csv(file_path)

    if shuffle_data:
        df = shuffle(df, random_state=42)
        # Resettiamo gli indici per pulizia dopo il mescolamento
        df.reset_index(drop=True, inplace=True)

    # 1. Estraiamo la colonna delle etichette (y) usando l'indice
    y = df.iloc[:, label_column_index].values

    # 2. Otteniamo le features (X) eliminando la colonna delle etichette
    # df.columns[label_column_index] trova il NOME della colonna a quell'indice
    nome_colonna_label = df.columns[label_column_index]
    X = df.drop(nome_colonna_label, axis=1).values

    print(f"-> Trovati {X.shape[0]} campioni con {X.shape[1]} features.")
    return X, y


# --- 2. FUNZIONE PER PLOTTARE TRAINING HISTORY ---
def plot_training_history(history):
    """
    Genera un grafico affiancato per Loss e Accuracy di Train e Validation.
    Prende in input l'oggetto 'history' restituito da model.fit() di Keras.
    """
    # Estraiamo le metriche dal dizionario history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoche = range(1, len(acc) + 1)

    # Creiamo una figura con due subplot affiancati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico Accuracy
    ax1.plot(epoche, acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epoche, val_acc, 'r--', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Andamento Accuracy', fontsize=14)
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Grafico Loss
    ax2.plot(epoche, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epoche, val_loss, 'r--', label='Validation Loss', linewidth=2)
    ax2.set_title('Andamento Loss (Errore)', fontsize=14)
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()


# --- 3. FUNZIONE PER PLOTTARE LA CONFUSION MATRIX ---
def plot_confusion_matrix(cm, class_names):
    """
    Plotta una matrice di confusione usando Seaborn per una visualizzazione elegante a mappa di calore.

    Parametri:
    - cm: La matrice di confusione calcolata (es. da sklearn.metrics.confusion_matrix).
    - class_names: Lista di stringhe con i nomi delle gesture (per le etichette degli assi).
    """
    plt.figure(figsize=(10, 8))

    # Creiamo la heatmap. 'annot=True' mostra i numeri dentro i quadrati.
    # 'fmt=d' assicura che i numeri siano interi e non scientifici.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Numero di predizioni'})

    plt.title('Matrice di Confusione', fontsize=16)
    plt.ylabel('Classe Reale (Ground Truth)', fontsize=12, fontweight='bold')
    plt.xlabel('Classe Predetta (Dal Modello)', fontsize=12, fontweight='bold')

    # Ruota le etichette sull'asse X per evitare che si sovrappongano se hai tante classi
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()