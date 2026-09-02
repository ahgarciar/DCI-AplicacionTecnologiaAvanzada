import CargaInstancia
import KNN_Algoritmo
import SplitInstacia
import MatrizConfusion
import statistics

instancia = None
id = 0

diccioMetricas = {
    0: [1, 2, 3, 5, 8, 9, 11],
}


instancia = CargaInstancia.cargarInstancia("../iris/iris.csv")
instance_name = "iris"

porc_Entrenamiento = 0.8
entrenamiento, prueba = SplitInstacia.split_instance(instancia, porc_Entrenamiento)

################################################################################
##Calcula valor de K
import math as m
k = m.sqrt(len(entrenamiento))  # valor inicial de prueba para K
k = int(k)
print("Valor de K: " + str(k))
################################################################################

metricas = diccioMetricas[id] #recupera todas las metricas para la instancia actual

for metrica in metricas:
    ejecuciones = 0
    tot_ejecuciones = 30
    resultados = []
    while ejecuciones<tot_ejecuciones:
        ################################################################################
        ##get respuestas por algoritmo
        respKnn = KNN_Algoritmo.probarKnn(entrenamiento, prueba, k, tipoMetrica=metrica)
        #print(respKnn)
        ##get respuestas reales
        respCorrectas = list(prueba["class"]) #list(prueba["Play Tennis"])
        #print(respCorrectas)
        ################################################################################
        ##calcula matriz de confusion
        resultado = MatrizConfusion.confusionMatrix(respCorrectas, respKnn, 3)
        resultado = round(resultado, 2)
        resultados.append(resultado)
        ################################################################################
        entrenamiento, prueba = SplitInstacia.split_instance(instancia, porc_Entrenamiento)
        ejecuciones +=1
    print(resultados)
    mediana = statistics.median(resultados)
    print(mediana)

