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
        X_train_pca, X_test_pca, _ = aplicar_pca(
            X_train,
            X_test,
            K
        )

        accuracy = regression_logistica(
            X_train_pca,
            y_train,
            X_test_pca,
            y_test
        )

        accuracies.append(accuracy)

        print(f"K = {K:3d} | Accuracy = {accuracy:.4f}")

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


#----------------- EJECUCIÓN DEL TP -----------------

def main():
    """
    Ejecuta los ejercicios 1.a, 1.b y 1.c del trabajo práctico.
    """

    #----------------- CARGA DEL DATASET -----------------

    X_train, y_train = cargar_datos(DATASET_DIR, "train")
    X_test, y_test = cargar_datos(DATASET_DIR, "test")

    print("Dimensiones del dataset:")
    print("Train:", X_train.shape)
    print("Test: ", X_test.shape)
    print()

    #----------------- EJERCICIO 1.A -----------------

    accuracy_sin_pca = regression_logistica(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("1-a) Accuracy sin PCA:", accuracy_sin_pca)
    print()

    #----------------- EJERCICIO 1.B -----------------

    K = 2

    X_train_pca, X_test_pca, _ = aplicar_pca(
        X_train,
        X_test,
        K
    )

    print("1-b) Antes de PCA:  ", X_test.shape)
    print("     Después de PCA:", X_test_pca.shape)
    print()

    graficar_pca(
        X_test_pca,
        y_test,
        "PCA - Conjunto de test"
    )

    #----------------- EJERCICIO 1.C -----------------

    valores_K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100, 150, 200]

    print("1-c) Evaluando PCA con diferentes valores de K:")

    accuracies_pca = evaluar_pca(
        X_train,
        y_train,
        X_test,
        y_test,
        valores_K
    )

    graficar_accuracy_pca(
        valores_K,
        accuracies_pca,
        accuracy_sin_pca
    )


if __name__ == "__main__":
    main()