# Librerías necesarias
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

# ─────────────────────────────────────────────────────────────
# GENERADOR DE CASOS DE USO
# Genera aleatoriamente dos DataFrames (train y test) con
# distribución sesgada y el output esperado del modelo
# ─────────────────────────────────────────────────────────────
def generar_caso_de_uso_transformar_y_ajustar_elasticnet():
    # Parámetros aleatorios: tamaños, hiperparámetros y nombre del target
    n_train    = random.randint(80, 150)
    n_test     = random.randint(20, 40)
    n_feat     = random.randint(3, 6)
    alpha      = round(random.uniform(0.01, 1.0), 3)
    l1_ratio   = round(random.uniform(0.1, 0.9), 2)
    target_col = random.choice(['precio', 'consumo', 'rendimiento'])
    feat_cols  = [f'x{i}' for i in range(n_feat)]

    # Genera datos con distribución exponencial (sesgada)
    coef_true = np.random.randn(n_feat)
    X_tr = np.random.exponential(scale=2.0, size=(n_train, n_feat))
    X_te = np.random.exponential(scale=2.0, size=(n_test,  n_feat))
    y_tr = X_tr @ coef_true + np.random.randn(n_train) * 0.5
    y_te = X_te @ coef_true + np.random.randn(n_test)  * 0.5

    # Construye los DataFrames
    df_train = pd.DataFrame(X_tr, columns=feat_cols)
    df_train[target_col] = y_tr
    df_test  = pd.DataFrame(X_te, columns=feat_cols)
    df_test[target_col]  = y_te

    # Introduce NaN en el target de train (~5%)
    nan_idx = random.sample(range(n_train), max(1, n_train // 20))
    df_train.loc[nan_idx, target_col] = np.nan

    input_data = {'df_train': df_train.copy(), 'df_test': df_test.copy(),
                  'target_col': target_col, 'alpha': alpha, 'l1_ratio': l1_ratio}

    # Calcula el output esperado (ground truth)
    df_tr_clean = df_train.dropna(subset=[target_col])
    X_train_arr = df_tr_clean[feat_cols].to_numpy()
    y_train_arr = df_tr_clean[target_col].to_numpy()
    X_test_arr  = df_test[feat_cols].to_numpy()
    y_test_arr  = df_test[target_col].to_numpy()

    n_q   = min(1000, len(X_train_arr))
    qt    = QuantileTransformer(n_quantiles=n_q, output_distribution='uniform', random_state=42)
    Xtr_t = qt.fit_transform(X_train_arr)
    Xte_t = qt.transform(X_test_arr)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
    model.fit(Xtr_t, y_train_arr)
    preds = model.predict(Xte_t)

    return input_data, {
        'predicciones': preds,
        'mae': round(float(mean_absolute_error(y_test_arr, preds)), 6),
        'coeficientes': model.coef_
    }


# ─────────────────────────────────────────────────────────────
# PASO 3 — Comprueba que el generador funciona correctamente
# ─────────────────────────────────────────────────────────────
i, o = generar_caso_de_uso_transformar_y_ajustar_elasticnet()

print('---- inputs ----')
for k, v in i.items():
    print('\n', k, ':\n', v)

print('\n---- expected output ----\n', o)
