from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#----------------- CONFIGURACIÓN -----------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_tp1"


#----------------- CARGA DE DATOS -----------------

def cargar_datos(carpeta_dataset, conjunto):
    """
    Carga las imágenes y etiquetas de train o test.

    Cada imagen de 128x128 se convierte en un vector de 16384 elementos.
    """

    carpeta_imagenes = carpeta_dataset / conjunto
    ruta_etiquetas = carpeta_dataset / f"{conjunto}_labels.csv"

    etiquetas = pd.read_csv(ruta_etiquetas)

    imagenes_vectorizadas = []

    for archivo in etiquetas["archivo"]:
        ruta_imagen = carpeta_imagenes / archivo

        with Image.open(ruta_imagen) as img:
            img = img.convert("L")
            img_array = np.array(img)

        # Imagen 128x128 -> vector de 16384 elementos
        img_vector = img_array.reshape(-1)
        imagenes_vectorizadas.append(img_vector)

    # Cada fila de X representa una imagen
    X = np.array(imagenes_vectorizadas)

    # Clase correspondiente a cada imagen
    y = etiquetas["clase"].to_numpy()

    return X, y


#----------------- REGRESIÓN LOGÍSTICA -----------------

def regression_logistica(X_train, y_train, X_test, y_test):
    """
    Entrena una regresión logística con los datos de entrenamiento
    y devuelve la accuracy obtenida sobre el conjunto de test.
    """

    modelo = LogisticRegression(max_iter=2000)

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def entrenar_regresion_logistica(X_train, y_train): # Utilizada en el ej 2
    """
    Entrena y devuelve un modelo de regresión logística.
    """

    modelo = LogisticRegression(max_iter=2000)
    modelo.fit(X_train, y_train)

    return modelo

#----------------- PCA -----------------

def aplicar_pca(X_train, X_test, K):
    """
    Ajusta PCA con K componentes sobre el conjunto de entrenamiento
    y transforma tanto train como test.
    """

    pca = PCA(n_components=K, random_state=42)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca


def graficar_pca(X_pca, y, titulo):
    """
    Grafica las dos primeras componentes principales
    separando las muestras según su clase.
    """

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_pca[y == 0, 0],
        X_pca[y == 0, 1],
        label="Normal",
        alpha=0.6
    )

    plt.scatter(
        X_pca[y == 1, 0],
        X_pca[y == 1, 1],
        label="Neumonía",
        alpha=0.6
    )

    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title(titulo)
    plt.legend()
    plt.grid()

    plt.show()


def evaluar_pca(X_train, y_train, X_test, y_test, valores_K):
    """
    Evalúa la regresión logística para distintos valores de K
    luego de aplicar PCA.
    """

    accuracies = []

    for K in valores_K:
        X_train_pca, X_test_pca, _ = aplicar_pca(X_train, X_test, K)

        accuracy = regression_logistica(X_train_pca, y_train, X_test_pca, y_test)

        accuracies.append(accuracy)

    return accuracies


def graficar_accuracy_pca(valores_K, accuracies_pca, accuracy_sin_pca):
    """
    Grafica la accuracy obtenida para cada valor de K
    y la compara con la accuracy del modelo sin PCA.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        valores_K,
        accuracies_pca,
        marker="o",
        label="PCA + Regresión Logística"
    )

    plt.axhline(
        y=accuracy_sin_pca,
        linestyle="--",
        label="Regresión Logística sin PCA"
    )

    plt.xlabel("Número de componentes principales K")
    plt.ylabel("Accuracy")
    plt.title("Accuracy en función de K")
    plt.legend()
    plt.grid()

    plt.show()


#----------------- EJ 2 -----------------

def perturbar_imagenes(X, p):
    """
    Rota 180 grados cada imagen con probabilidad p.
    """

    X_perturbado = X.copy()

    for i in range(len(X_perturbado)):

        if np.random.rand() < p:
            imagen = X_perturbado[i].reshape(128, 128)

            imagen_rotada = np.rot90(imagen, 2)

            X_perturbado[i] = imagen_rotada.reshape(-1)

    return X_perturbado


def graficar_pca_perturbado(X_test, y_test, pca, valores_p):
    """
    Genera una realización perturbada para cada valor de p
    y grafica las dos primeras componentes principales.
    """

    for p in valores_p:
        X_test_perturbado = perturbar_imagenes(X_test, p)

        X_test_pca = pca.transform(X_test_perturbado)

        graficar_pca(X_test_pca, y_test, f"PCA - Test perturbado con p = {p}")

#----------------- MONTE CARLO -----------------

def simulacion_monte_carlo(X_test, y_test, pca, modelo, p, NMC):
    """
    Realiza NMC simulaciones perturbando las imágenes de test
    con probabilidad p y devuelve las accuracies obtenidas.
    """

    accuracies = []

    for _ in range(NMC):
        X_test_perturbado = perturbar_imagenes(X_test, p)

        X_test_pca = pca.transform(X_test_perturbado)

        y_pred = modelo.predict(X_test_pca)

        accuracy = accuracy_score(y_test, y_pred)

        accuracies.append(accuracy)

    return accuracies


def evaluar_accuracy_monte_carlo(X_test, y_test, pca, modelo, valores_p, NMC):
    """
    Estima el accuracy medio para cada valor de p
    mediante simulación Monte Carlo.
    """

    accuracies_medias = []
    accuracies_por_p = []

    for p in valores_p:

        accuracies = simulacion_monte_carlo(X_test, y_test, pca, modelo, p, NMC)

        accuracy_media = np.mean(accuracies)
        accuracies_medias.append(accuracy_media)
        accuracies_por_p.append(accuracies)

    return accuracies_medias, accuracies_por_p


def graficar_accuracy_monte_carlo(valores_p, accuracies_medias):
    """
    Grafica el accuracy medio estimado en función de p.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(valores_p, accuracies_medias, marker="o")

    plt.xlabel("Probabilidad de perturbación p")
    plt.ylabel("Accuracy medio")
    plt.title("Accuracy medio en función de p")
    plt.grid()

    plt.show()


def calcular_probabilidad_perdida(accuracies_por_p, accuracy_original, delta):
    """
    Estima la probabilidad de que la pérdida supere delta.
    """

    probabilidades = []

    for accuracies in accuracies_por_p:
        perdidas = accuracy_original - np.array(accuracies)

        probabilidad = np.mean(perdidas > delta)
        probabilidades.append(probabilidad)

    return probabilidades


def graficar_probabilidad_perdida(valores_p, probabilidades_perdida, delta):
    """
    Grafica la probabilidad de que la pérdida supere delta.
    """

    plt.figure(figsize=(8, 6))
    plt.plot(valores_p, probabilidades_perdida, marker="o")

    plt.xlabel("Probabilidad de perturbación p")
    plt.ylabel("Probabilidad estimada")
    plt.title(f"Probabilidad de L(p) > {delta}")
    plt.grid()

    plt.show()


def graficar_histogramas_accuracies(valores_p, accuracies_por_p):
    """
    Grafica los histogramas de accuracies para cada valor de p.
    """

    for p, accuracies in zip(valores_p, accuracies_por_p):
        plt.figure(figsize=(8, 6))
        plt.hist(accuracies, bins=10)
        plt.xlabel("Accuracy")
        plt.ylabel("Frecuencia")
        plt.title(f"Histograma de accuracies para p = {p}")
        plt.grid()

        plt.show()

#----------------- EJECUCIÓN DEL TP -----------------

def main():
    """
    Ejecuta los ejercicios del trabajo práctico.
    """

    # Semilla para que las simulaciones aleatorias sean reproducibles
    np.random.seed(42)

    #----------------- CARGA DEL DATASET -----------------

    X_train, y_train = cargar_datos(DATASET_DIR, "train")
    X_test, y_test = cargar_datos(DATASET_DIR, "test")

    print("Dimensiones del dataset:")
    print("Train:", X_train.shape)
    print("Test: ", X_test.shape)
    print()


    #----------------- EJERCICIO 1.A -----------------

    accuracy_sin_pca = regression_logistica(X_train, y_train, X_test, y_test)

    print("1-a) Accuracy sin PCA:", accuracy_sin_pca)
    print()


    #----------------- EJERCICIO 1.B -----------------

    K_visualizacion = 2

    X_train_pca, X_test_pca, pca = aplicar_pca(X_train, X_test, K_visualizacion)

    print("1-b) Antes de PCA:  ", X_test.shape)
    print("     Después de PCA:", X_test_pca.shape)
    print()

    graficar_pca(X_test_pca, y_test, "PCA - Conjunto de test")


    #----------------- EJERCICIO 1.C -----------------

    valores_K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100, 150, 200]

    accuracies_pca = evaluar_pca(X_train, y_train, X_test, y_test, valores_K)

    print("1-c) Accuracies para cada K:")

    for k, accuracy in zip(valores_K, accuracies_pca):
        print(f"K = {k:3d} | Accuracy = {accuracy:.4f}")

    print()

    graficar_accuracy_pca(valores_K, accuracies_pca, accuracy_sin_pca)


    #----------------- EJERCICIO 2 -----------------

    valores_p = [0.1, 0.3, 0.5, 0.7, 0.9]

    K_MC = 2

    # Se entrena PCA y la regresión logística una única vez usando train sin perturbar
    X_train_pca_mc, X_test_pca_mc, pca_mc = aplicar_pca(X_train, X_test, K_MC)
    modelo_pca_mc = entrenar_regresion_logistica(X_train_pca_mc, y_train)

    y_pred_pca = modelo_pca_mc.predict(X_test_pca_mc)
    accuracy_original = accuracy_score(y_test, y_pred_pca)

    print("Accuracy original PCA + LR con K = 2:", accuracy_original)
    print()


    #----------------- EJERCICIO 2.A -----------------

    print("2-a) Graficando PCA para distintos valores de p...")
    print()

    graficar_pca_perturbado(X_test, y_test, pca_mc, valores_p)


    #----------------- EJERCICIO 2.B -----------------

    NMC = 1000

    print("2-b) Calculando accuracy medio estimado mediante Monte Carlo...")

    accuracies_medias, accuracies_por_p = evaluar_accuracy_monte_carlo(X_test, y_test, pca_mc, modelo_pca_mc, valores_p, NMC)

    for p, accuracy_media in zip(valores_p, accuracies_medias):
        print(f"p = {p:.1f} | E[Ap] = {accuracy_media:.4f}")

    print()

    graficar_accuracy_monte_carlo(valores_p, accuracies_medias)


    #----------------- EJERCICIO 2.C -----------------

    delta = 0.1

    probabilidades_perdida = calcular_probabilidad_perdida(accuracies_por_p, accuracy_original, delta)

    print("2-c) Probabilidad estimada de pérdida mayor a delta:")

    for p, probabilidad in zip(valores_p, probabilidades_perdida):
        print(f"p = {p:.1f} | P(L(p) > {delta}) = {probabilidad:.4f}")

    print()

    graficar_probabilidad_perdida(valores_p, probabilidades_perdida, delta)


    #----------------- EJERCICIO 2.D -----------------

    graficar_histogramas_accuracies(valores_p, accuracies_por_p)


if __name__ == "__main__":
    main()
