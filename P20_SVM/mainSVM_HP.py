import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, ParameterSampler, ParameterGrid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from joblib import dump

# "random", "grid" o "heuristica"
metodo_busqueda = "heuristica"

#
#SE REQUIERE:
# 1.- ESTABLECER LOS HIPERPARAMETROS QUE SE MODIFICARAN DEL MODELO
# 2.- DEFINIR EL ESPACIO DE SOLUCIONES DE BUSQUEDA DE LOS HIPERPARAMETROS
#

# CARGA DE INSTANCIA
instancia = pd.read_csv("../iris/iris.csv")

# SEPARA LA INSTANCIA
X = instancia.iloc[:, :-1].copy()
y = instancia.iloc[:, -1].copy()   # nominal

# CODIFICA LAS CLASES
encoder = LabelEncoder()
y_int = encoder.fit_transform(y)

n_clases = len(encoder.classes_)
n_feats = X.shape[1]
print(f"n_feats={n_feats} | n_clases={n_clases} | clases={list(encoder.classes_)}")

# SPLIT 70/15/15 con estratificación
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_int, test_size=0.30, random_state=7, stratify=y_int
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=7, stratify=y_temp
)

# ESCALADO (con TRAIN)
x_scaler = StandardScaler().fit(X_train)
X_train_s = x_scaler.transform(X_train)
X_val_s   = x_scaler.transform(X_val)
X_test_s  = x_scaler.transform(X_test)

# MODELO SVM
params_grid = [
    # Kernel lineal
    {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 1, 10, 100] # -> C [ 0.01 - 100 ]
    },
    # Kernel RBF (gaussiano)
    {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 1, 0.1, 0.01, 0.001]
    },
    # Kernel polinómico
    {
        "kernel": ["poly"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 1, 0.1, 0.01],
        "degree": [2, 3, 4, 5],  # típicamente 2 o 3
        "coef0": [0.0, 0.5, 1.0]  # término independiente del polinomio
    }
]

# AJUSTE DE HIPERPARÁMETROS CON TRAIN Y VAL
def evaluar_parametros(params):
    modelo = SVC(**params, random_state=7) # SVC(kernel="rbf", C=1, gamma="scale", random_state=7)
    modelo.fit(X_train_s, y_train) #entrenamiento del modelo
    predicciones = modelo.predict(X_val_s) # prueba del modelo
    return accuracy_score(y_val, predicciones) #calcular la metrica de precision


if metodo_busqueda == "random":
    # Random Search -> prueba 25 combinaciones distintas al azar
    candidatos = ParameterSampler(
        params_grid,
        n_iter=25,
        random_state=7
    )
    mejores_params = max(candidatos, key=evaluar_parametros)

elif metodo_busqueda == "grid":
    # Grid Search -> prueba las 269 combinaciones de params_grid
    candidatos = ParameterGrid(params_grid)
    mejores_params = max(candidatos, key=evaluar_parametros)

elif metodo_busqueda == "heuristica":
    # Heurística básica con un RBF de base
    mejores_params = {"kernel": "rbf", "C": 1, "gamma": "scale"} #solucion inicial de referencia
    mejor_acc = evaluar_parametros(mejores_params)

    # Primero ajusta C y después gamma. Solo acepta mejoras
    for parametro in ("C", "gamma"): #busqueda local con best improvement
        for valor in params_grid[1][parametro]:
            if valor == mejores_params[parametro]:
                continue

            candidato = {**mejores_params, parametro: valor}
            acc = evaluar_parametros(candidato)

            if acc > mejor_acc:
                mejores_params = candidato
                mejor_acc = acc

else:
    raise ValueError('Usa "random", "grid" o "heuristica".')

print(f"\nMétodo de búsqueda: {metodo_busqueda}")
print(f"Mejores hiperparámetros: {mejores_params}")

# Entrenamiento con TRAIN
svm_model = SVC(
    **mejores_params,
    probability=True,
    random_state=7
)
svm_model.fit(X_train_s, y_train)

# EVALUACIÓN EN VALIDACIÓN
val_pred = svm_model.predict(X_val_s)
val_acc  = accuracy_score(y_val, val_pred)
print(f"\nValidación - Accuracy: {val_acc:.4f}")

# REENTRENAMIENTO con TRAIN + VAL (recomendado antes de TEST)
X_trainval = np.vstack([X_train_s, X_val_s])
y_trainval = np.concatenate([y_train, y_val])

svm_final = SVC(
    **mejores_params,
    probability=True,
    random_state=7
)
svm_final.fit(X_trainval, y_trainval)

# EVALUACIÓN FINAL EN TEST
y_prob = svm_final.predict_proba(X_test_s)  # probabilities
y_pred = svm_final.predict(X_test_s)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest - Accuracy: {test_acc:.4f}")

print("\nClassification report (TEST):")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm_df)

# Guarda el modelo
dump(svm_final, "svm_model.joblib")
dump(x_scaler,  "x_scaler.joblib")
dump(encoder,   "label_encoder.joblib")