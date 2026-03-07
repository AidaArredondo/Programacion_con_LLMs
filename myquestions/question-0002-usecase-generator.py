import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_pivotar_y_calcular_ranking():
    import pandas as pd
    import random
    from sklearn.preprocessing import StandardScaler

    n_exp        = random.randint(4, 8)
    exp_col      = random.choice(['experimento', 'modelo', 'config'])
    metrica_col  = random.choice(['metrica', 'indicador', 'kpi'])
    valor_col    = random.choice(['valor', 'score', 'resultado'])
    metricas     = random.sample(['accuracy', 'f1', 'precision', 'recall', 'auc', 'rmse'], 3)
    metrica_objetivo = metricas[0]

    rows = []
    for i in range(n_exp):
        for met in metricas:
            rows.append({exp_col: f'exp_{i}', metrica_col: met,
                         valor_col: round(random.uniform(0.5, 1.0), 4)})

    df = pd.DataFrame(rows)
    input_data = {'df': df.copy(), 'exp_col': exp_col, 'metrica_col': metrica_col,
                  'valor_col': valor_col, 'metrica_objetivo': metrica_objetivo}

    pivot = pd.pivot_table(df, values=valor_col, index=exp_col,
                           columns=metrica_col, aggfunc='mean')
    res = pivot[[metrica_objetivo]].copy()
    scaler = StandardScaler()
    res['score_normalizado'] = scaler.fit_transform(res[[metrica_objetivo]])
    res = res.sort_values('score_normalizado', ascending=False)
    res['ranking'] = range(1, len(res) + 1)
    res = res.reset_index()[[exp_col, metrica_objetivo, 'score_normalizado', 'ranking']]

    return input_data, res


def pivotar_y_calcular_ranking(df, exp_col, metrica_col, valor_col, metrica_objetivo):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    pivot = pd.pivot_table(df, values=valor_col, index=exp_col,
                           columns=metrica_col, aggfunc='mean')
    res = pivot[[metrica_objetivo]].copy()
    scaler = StandardScaler()
    res['score_normalizado'] = scaler.fit_transform(res[[metrica_objetivo]])
    res = res.sort_values('score_normalizado', ascending=False)
    res['ranking'] = range(1, len(res) + 1)
    return res.reset_index()[[exp_col, metrica_objetivo, 'score_normalizado', 'ranking']]
