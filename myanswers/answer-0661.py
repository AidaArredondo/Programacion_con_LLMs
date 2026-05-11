import pandas as pd

def limpiar_dataframe(df, umbral):
    # Paso 1: eliminar duplicados
    result = df.drop_duplicates()

    # Paso 2: eliminar filas con más NaNs que el umbral
    result = result[result.isna().sum(axis=1) <= umbral]

    # Paso 3: rellenar NaNs restantes con el promedio de cada columna
    for col in result.columns:
        col_mean = result[col].mean()
        result[col] = result[col].fillna(col_mean)

    # Paso 4: reiniciar índice
    return result.reset_index(drop=True)
