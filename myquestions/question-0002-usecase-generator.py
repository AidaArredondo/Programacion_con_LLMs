# Librerías necesarias
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# GENERADOR DE CASOS DE USO
# Genera aleatoriamente un DataFrame de resultados de
# experimentos junto con el ranking esperado
# ─────────────────────────────────────────────────────────────
def generar_caso_de_uso_pivotar_y_calcular_ranking():
    # Parámetros aleatorios: número de experimentos y nombres de columnas
    n_exp        = random.randint(4, 8)
    exp_col      = random.choice(['experimento', 'modelo', 'config'])
    metrica_col  = random.choice(['metrica', 'indicador', 'kpi'])
    valor_col    = random.choice(['valor', 'score', 'resultado'])

    # Elige 3 métricas al azar y una de ellas será la objetivo
    metricas         = random.sample(['accuracy', 'f1', 'precision', 'recall', 'auc', 'rmse'], 3)
    metrica_objetivo = metricas[0]

    # Construye el DataFrame con un valor aleatorio por experimento y métrica
    rows = []
    for i in range(n_exp):
        for met in metricas:
            rows.append({exp_col: f'exp_{i}', metrica_col: met,
                         valor_col: round(random.uniform(0.5, 1.0), 4)})
    df = pd.DataFrame(rows)
    input_data = {'df': df.copy(), 'exp_col': exp_col, 'metrica_col': metrica_col,
                  'valor_col': valor_col, 'metrica_objetivo': metrica_objetivo}

    # Calcula el output esperado (ground truth)
    pivot = pd.pivot_table(df, values=valor_col, index=exp_col,
                           columns=metrica_col, aggfunc='mean')
    res = pivot[[metrica_objetivo]].copy()
    scaler = StandardScaler()
    res['score_normalizado'] = scaler.fit_transform(res[[metrica_objetivo]])
    res = res.sort_values('score_normalizado', ascending=False)
    res['ranking'] = range(1, len(res) + 1)
    res = res.reset_index()[[exp_col, metrica_objetivo, 'score_normalizado', 'ranking']]

    return input_data, res


# ─────────────────────────────────────────────────────────────
# PASO 3 — Comprueba que el generador funciona correctamente
# ─────────────────────────────────────────────────────────────
i, o = generar_caso_de_uso_pivotar_y_calcular_ranking()

print('---- inputs ----')
for k, v in i.items():
    print('\n', k, ':\n', v)

print('\n---- expected output ----\n', o)
