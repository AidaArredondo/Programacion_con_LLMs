import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_reconstruir_serie_temporal():
    import pandas as pd
    import numpy as np
    import random
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    n = random.randint(15, 35)
    fecha_col = random.choice(['timestamp', 'fecha', 'datetime_utc'])
    valor_col = random.choice(['temperatura', 'presion', 'humedad'])

    base = pd.date_range('2024-01-01', periods=n, freq='h')
    fechas = list(base)
    for _ in range(max(1, n // 10)):
        fechas.append(fechas[random.randint(0, n - 1)])
    random.shuffle(fechas)

    valores = np.random.uniform(15, 35, size=len(fechas)).round(2).tolist()
    for i in random.sample(range(len(valores)), max(1, len(valores) // 7)):
        valores[i] = np.nan

    df = pd.DataFrame({fecha_col: fechas, valor_col: valores})
    input_data = {'df': df.copy(), 'fecha_col': fecha_col, 'valor_col': valor_col}

    df_g = df.copy()
    df_g[fecha_col] = pd.to_datetime(df_g[fecha_col])
    df_g = df_g.drop_duplicates(subset=[fecha_col], keep='first')
    df_g = df_g.sort_values(fecha_col).reset_index(drop=True)
    df_g[valor_col] = df_g[valor_col].interpolate(method='linear')
    imp = SimpleImputer(strategy='mean')
    df_g[valor_col] = imp.fit_transform(df_g[[valor_col]])
    scaler = StandardScaler()
    df_g['valor_escalado'] = scaler.fit_transform(df_g[[valor_col]])

    return input_data, df_g


def reconstruir_serie_temporal(df, fecha_col, valor_col):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.drop_duplicates(subset=[fecha_col], keep='first')
    df = df.sort_values(fecha_col).reset_index(drop=True)
    df[valor_col] = df[valor_col].interpolate(method='linear')
    imp = SimpleImputer(strategy='mean')
    df[valor_col] = imp.fit_transform(df[[valor_col]])
    scaler = StandardScaler()
    df['valor_escalado'] = scaler.fit_transform(df[[valor_col]])
    return df
