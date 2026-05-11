import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def analizar_error_por_cuartil(X, y, n_cuartiles=4):
    # 1. Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Escalar (fit solo en train)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 3. Entrenar Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train_sc, y_train)

    # 4. Errores absolutos por muestra
    y_pred = model.predict(X_test_sc)
    abs_errors = np.abs(y_test - y_pred)

    # 5. Límites de cuartiles sobre y_test
    percentiles = np.linspace(0, 100, n_cuartiles + 1)
    limits = np.percentile(y_test, percentiles)

    # 6. MAE por segmento
    rows = []
    for i in range(n_cuartiles):
        lo, hi = limits[i], limits[i + 1]
        if i == 0:
            mask = (y_test >= lo) & (y_test <= hi)
        else:
            mask = (y_test > lo) & (y_test <= hi)
        mae_q = float(np.mean(abs_errors[mask])) if mask.sum() > 0 else float('nan')
        rows.append({'cuartil': i + 1, 'mae_promedio': mae_q})

    # 7. DataFrame ordenado
    return pd.DataFrame(rows).sort_values('cuartil').reset_index(drop=True)
