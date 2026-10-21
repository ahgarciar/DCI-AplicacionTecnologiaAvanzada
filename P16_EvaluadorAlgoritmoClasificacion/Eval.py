
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    jaccard_score,
    matthews_corrcoef,
    cohen_kappa_score,
    hamming_loss,
    zero_one_loss,
    roc_auc_score,
    log_loss
)

# 1. DATOS REALES Y PREDICCIONES
# Ejemplo con tres clases
y_test = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] #real
y_pred = [0, 1, 1, 0, 2, 2, 0, 1, 2, 1, 1, 2] #predicha por el algoritmo

y_pred1 = [0, 1, 1, 0, 2, 2, 0, 1, 2, 1, 1, 2] #predicha por el algoritmo
y_pred2 = [0, 1, 1, 0, 2, 2, 0, 1, 2, 1, 1, 2] #predicha por el algoritmo
y_pred3 = [0, 1, 1, 0, 2, 2, 0, 1, 2, 1, 1, 2] #predicha por el algoritmo

clases = np.unique(np.concatenate([y_test, y_pred]))

# 2. MATRIZ DE CONFUSIÓN
matriz = confusion_matrix(
    y_test, y_pred, labels=clases
)

print("MATRIZ DE CONFUSIÓN")
print(matriz)

plt.figure(figsize=(7, 5))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=clases,
    yticklabels=clases
)

plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.title("Matriz de confusión")
plt.tight_layout()
plt.show()

# 3. MÉTRICAS GENERALES
print("\nMÉTRICAS GENERALES")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Balanced Accuracy:",
      balanced_accuracy_score(y_test, y_pred))

print("MCC:",
      matthews_corrcoef(y_test, y_pred))

print("Cohen Kappa:",
      cohen_kappa_score(y_test, y_pred))

print("Hamming Loss:",
      hamming_loss(y_test, y_pred))

print("Zero-One Loss:",
      zero_one_loss(y_test, y_pred))


#¿Qué significa Macro, Micro y Weighted?
#Estas tres formas de promediar son especialmente importantes cuando existen tres o más clases.

# Macro: calcula la métrica individualmente para cada clase y obtiene su promedio,
#           otorgando la misma importancia a todas las clases.

# Micro: acumula los TP, FP y FN de todas las clases antes de calcular la métrica.
#           En clasificación multiclase de etiqueta única, precision, recall y F1 micro coinciden con accuracy.

# Weighted: calcula las métricas por clase y obtiene un promedio ponderado
#           según la cantidad de instancias reales de cada clase.


# 4. PRECISIÓN, RECALL Y F1

print("\nMÉTRICAS MACRO")

print("Precision:",
      precision_score(
          y_test, y_pred,
          average="macro",
          zero_division=0
      ))

print("Recall / Sensibilidad:",
      recall_score(
          y_test, y_pred,
          average="macro",
          zero_division=0
      ))

print("F1-Score:",
      f1_score(
          y_test, y_pred,
          average="macro",
          zero_division=0
      ))

print("F2-Score:",
      fbeta_score(
          y_test, y_pred,
          beta=2,
          average="macro",
          zero_division=0
      ))

print("Jaccard:",
      jaccard_score(
          y_test, y_pred,
          average="macro",
          zero_division=0
      ))

# 5. PROMEDIOS WEIGHTED Y MICRO

for promedio in ["micro", "weighted"]:

    print(f"\nPROMEDIO {promedio.upper()}")

    print("Precision:",
          precision_score(
              y_test, y_pred,
              average=promedio,
              zero_division=0
          ))

    print("Recall:",
          recall_score(
              y_test, y_pred,
              average=promedio,
              zero_division=0
          ))

    print("F1:",
          f1_score(
              y_test, y_pred,
              average=promedio,
              zero_division=0
          ))

# 6. REPORTE POR CLASE

print("\nREPORTE DE CLASIFICACIÓN")

print(classification_report(
    y_test,
    y_pred,
    labels=clases,
    zero_division=0
))

# 7. ESPECIFICIDAD POR CLASE

print("\nESPECIFICIDAD POR CLASE")

for i, clase in enumerate(clases):

    TP = matriz[i, i]
    FN = matriz[i, :].sum() - TP
    FP = matriz[:, i].sum() - TP
    TN = matriz.sum() - TP - FN - FP

    sensibilidad = (
        TP / (TP + FN)
        if (TP + FN) > 0 else 0
    )

    especificidad = (
        TN / (TN + FP)
        if (TN + FP) > 0 else 0
    )

    print(f"Clase {clase}")
    print(f"  Sensibilidad: {sensibilidad:.4f}")
    print(f"  Especificidad: {especificidad:.4f}")


#Métrica	                ¿Qué evalúa?	                                        Valor deseable
#Accuracy	                Proporción de predicciones correctas	                Cercano a 1
#Balanced Accuracy	        Promedio del recall de las clases	                    Cercano a 1
#Precision	                Proporción de positivos predichos que son correctos	    Cercano a 1
#Recall / Sensibilidad	    Proporción de positivos reales identificados	        Cercano a 1
#Especificidad	            Proporción de negativos reales identificados	        Cercano a 1
#F1-Score	                Media armónica de precision y recall	                Cercano a 1
#F2-Score	                Similar al F1, pero prioriza recall	                    Cercano a 1
#Jaccard	                Coincidencia entre positivos reales y predichos	        Cercano a 1
#MCC	                    Correlación entre predicciones y valores reales	        Cercano a 1
#Cohen's Kappa	            Concordancia corregida por azar	                        Cercano a 1
#Hamming Loss	            Proporción de clasificaciones incorrectas	            Cercano a 0
#Zero-One Loss	            Proporción de instancias mal clasificadas	            Cercano a 0
