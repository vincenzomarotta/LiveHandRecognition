# load_and_plot.py aggiornato

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import os
import numpy as np


def load_dataset(file_path, label_column_index=0, shuffle_data=True):
    print(f"Caricamento dati da: {file_path}...")
    df = pd.read_csv(file_path)
    if shuffle_data:
        df = shuffle(df, random_state=42)
        df.reset_index(drop=True, inplace=True)
    y = df.iloc[:, label_column_index].values
    nome_colonna_label = df.columns[label_column_index]
    X = df.drop(nome_colonna_label, axis=1).values
    print(f"-> Trovati {X.shape[0]} campioni con {X.shape[1]} features.")
    return X, y


def plot_training_history(history, save_path=None):
    """
    Plotta e opzionalmente salva i grafici di training/validation.
    save_path: se fornito (es. "plots/cnn_history.png"), salva il file invece di mostrarlo.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoche = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epoche, acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epoche, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Andamento Accuracy', fontsize=14)
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2.plot(epoche, loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epoche, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Andamento Loss', fontsize=14)
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Grafico training salvato in: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plotta e opzionalmente salva la confusion matrix.
    save_path: se fornito (es. "plots/cnn_cm.png"), salva il file.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Numero di predizioni'})
    plt.title('Matrice di Confusione', fontsize=16)
    plt.ylabel('Classe Reale', fontsize=12, fontweight='bold')
    plt.xlabel('Classe Predetta', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Confusion matrix salvata in: {save_path}")
    else:
        plt.show()
    plt.close()


def save_metrics_csv(all_labels, all_preds, class_names, avg_latency_ms,
                     output_csv="metrics_summary.csv"):
    """
    Salva un CSV con:
    - Una riga per classe: precision, recall, f1, support
    - Una riga 'macro avg' e 'weighted avg'
    - Una riga 'overall' con accuracy globale e latenza media

    Parametri:
    - all_labels:      lista/array delle etichette reali
    - all_preds:       lista/array delle predizioni
    - class_names:     lista dei nomi delle classi (in ordine di indice)
    - avg_latency_ms:  latenza media per immagine in ms
    - output_csv:      percorso del file CSV di output
    """
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    rows = []

    # Riga per ogni classe
    for class_name in class_names:
        if class_name in report:
            r = report[class_name]
            rows.append({
                'class': class_name,
                'precision': round(r['precision'], 4),
                'recall': round(r['recall'], 4),
                'f1_score': round(r['f1-score'], 4),
                'support': int(r['support']),
                'avg_latency_ms': ''  # solo nella riga overall
            })

    # Righe aggregate
    for avg_key in ('macro avg', 'weighted avg'):
        r = report[avg_key]
        rows.append({
            'class': avg_key,
            'precision': round(r['precision'], 4),
            'recall': round(r['recall'], 4),
            'f1_score': round(r['f1-score'], 4),
            'support': int(r['support']),
            'avg_latency_ms': ''
        })

    # Riga overall: accuracy + latenza media
    rows.append({
        'class': 'overall',
        'precision': '',
        'recall': '',
        'f1_score': '',
        'support': len(all_labels),
        'avg_latency_ms': round(avg_latency_ms, 4)
    })

    # Aggiungiamo anche l'accuracy globale nella riga overall
    rows[-1]['precision'] = round(report['accuracy'], 4)  # riuso precision come accuracy

    df = pd.DataFrame(rows)
    # Rinomina precision→accuracy nella riga overall per chiarezza
    df.to_csv(output_csv, index=False)
    print(f"[OK] Metriche aggregate salvate in: {output_csv}")
    return df