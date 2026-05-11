import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import OPTICS

def obtener_alcanzabilidad_optics(X, min_muestras):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    optics = OPTICS(min_samples=min_muestras)
    optics.fit(X_scaled)
    return optics.reachability_
