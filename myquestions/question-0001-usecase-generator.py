# Librerías necesarias
import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# GENERADOR DE CASOS DE USO
# Genera aleatoriamente un DataFrame sucio (con NaN, duplicados
# y desordenado) junto con el output esperado ya procesado
# ─────────────────────────────────────────────────────────────
def generar_caso_de_uso_reconstruir_serie_temporal():
    # Parámetros aleatorios: tamaño, nombre de columnas
    n = random.randint(15, 35)
    fecha_col = random.choice(['timestamp', 'fecha', 'datetime_utc'])
    valor_col = random.choice(['temperatura', 'presion', 'humedad'])

    # Genera timestamps y añade duplicados aleatoriamente
    base = pd.date_range('2024-01-01', periods=n, freq='h')
    fechas = list(base)
    for _ in range(max(1, n // 10)):
        fechas.append(fechas[random.randint(0, n - 1)])
    random.shuffle(fechas)  # desordena las fechas

    # Genera valores y añade NaN aleatoriamente (~15%)
    valores = np.random.uniform(15, 35, size=len(fechas)).round(2).tolist()
    for i in random.sample(range(len(valores)), max(1, len(valores) // 7)):
        valores[i] = np.nan

    # Construye el DataFrame sucio
    df = pd.DataFrame({fecha_col: fechas, valor_col: valores})
    input_data = {'df': df.copy(), 'fecha_col': fecha_col, 'valor_col': valor_col}

    # Calcula el output esperado (ground truth)
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

# ─────────────────────────────────────────────────────────────
# FUNCIÓN SOLUCIÓN
# Recibe el DataFrame sucio y devuelve uno limpio y escalado
# ─────────────────────────────────────────────────────────────
def reconstruir_serie_temporal(df, fecha_col, valor_col):
    df = df.copy()

    # Paso 1: convierte la columna de fecha a datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Paso 2: elimina filas duplicadas por fecha
    df = df.drop_duplicates(subset=[fecha_col], keep='first')

    # Paso 3: ordena por fecha y reinicia el índice
    df = df.sort_values(fecha_col).reset_index(drop=True)

    # Paso 4: rellena NaN con interpolación lineal
    df[valor_col] = df[valor_col].interpolate(method='linear')

    # Paso 5: imputa NaN residuales de los extremos con la media
    imp = SimpleImputer(strategy='mean')
    df[valor_col] = imp.fit_transform(df[[valor_col]])

    # Paso 6: escala los valores a media 0 y std 1
    scaler = StandardScaler()
    df['valor_escalado'] = scaler.fit_transform(df[[valor_col]])

    return df

# ─────────────────────────────────────────────────────────────
# PASO 3 — Comprueba que el generador funciona correctamente
# ─────────────────────────────────────────────────────────────
i, o = generar_caso_de_uso_reconstruir_serie_temporal()

print('---- inputs ----')
for k, v in i.items():
    print('\n', k, ':\n', v)

print('\n---- expected output ----\n', o)
