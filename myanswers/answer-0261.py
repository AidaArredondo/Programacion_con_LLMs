from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

def clasificar_jugadores(df, target_col):
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Imputar NaNs con KNN
    imputer = KNNImputer()
    X_imputed = imputer.fit_transform(X)

    # Escalar con RobustScaler (robusto a outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Entrenar regresión logística balanceada
    model = LogisticRegression(class_weight='balanced', max_iter=200)
    model.fit(X_scaled, y)

    return model, model.predict(X_scaled)
