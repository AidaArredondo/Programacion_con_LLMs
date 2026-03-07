# Librerías necesarias
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AgglomerativeClustering

# ─────────────────────────────────────────────────────────────
# GENERADOR DE CASOS DE USO
# Genera aleatoriamente un DataFrame con clases bien separadas
# y calcula la pureza esperada por fold
# ─────────────────────────────────────────────────────────────
def generar_caso_de_uso_validar_con_kfold_estratificado():
    # Parámetros aleatorios: clases, muestras, features, folds
    n_classes   = random.randint(2, 4)
    n_per_class = random.randint(25, 45)
    n_features  = random.randint(3, 5)
    n_clusters  = n_classes  # clusters == clases para que la pureza sea significativa
    n_splits    = random.choice([3, 4, 5])
    label_col   = random.choice(['categoria', 'clase', 'grupo'])
    feat_cols   = [f'feat_{i}' for i in range(n_features)]

    # Genera datos bien separados (cada clase centrada en i*3)
    X = np.vstack([np.random.randn(n_per_class, n_features) + i * 3
                   for i in range(n_classes)])
    y = np.hstack([np.full(n_per_class, i) for i in range(n_classes)])

    # Mezcla aleatoriamente las muestras
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # Construye el DataFrame
    df = pd.DataFrame(X, columns=feat_cols)
    df[label_col] = y.astype(int)

    input_data = {'df': df.copy(), 'feature_cols': feat_cols,
                  'label_col': label_col, 'n_clusters': n_clusters, 'n_splits': n_splits}

    # Calcula el output esperado (ground truth)
    X_arr = df[feat_cols].to_numpy()
    y_arr = df[label_col].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    purezas = []
    for _, test_idx in skf.split(X_arr, y_arr):
        X_te, y_te = X_arr[test_idx], y_arr[test_idx]
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_te)
        aciertos = sum(np.bincount(y_te[labels == c].astype(int)).max()
                       for c in np.unique(labels))
        purezas.append(aciertos / len(y_te))

    return input_data, {
        'pureza_por_fold': purezas,
        'pureza_media': round(float(np.mean(purezas)), 4),
        'pureza_std':   round(float(np.std(purezas)),  4)
    }


# ─────────────────────────────────────────────────────────────
# PASO 3 — Comprueba que el generador funciona correctamente
# ─────────────────────────────────────────────────────────────
i, o = generar_caso_de_uso_validar_con_kfold_estratificado()

print('---- inputs ----')
for k, v in i.items():
    print('\n', k, ':\n', v)

print('\n---- expected output ----\n', o)
