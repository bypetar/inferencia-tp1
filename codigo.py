from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Ubicacion base del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Dataset dentro del proyecto
carpeta_dataset = BASE_DIR / "dataset_tp1"


def cargar_datos(carpeta_dataset, conjunto):

    # Carpeta de imagenes: train o test
    carpeta_imagenes = carpeta_dataset / conjunto

    # Archivo CSV: train_labels.csv o test_labels.csv
    etiquetas = pd.read_csv(carpeta_dataset / f"{conjunto}_labels.csv")

    imagenes_vectorizadas = []

    for archivo in etiquetas["archivo"]:

        ruta_imagen = carpeta_imagenes / archivo

        img = Image.open(ruta_imagen).convert("L")
        img_array = np.array(img)

        img_vector = img_array.reshape(-1) # Transformar imagen 128x128 en vector de 16384 elementos

        imagenes_vectorizadas.append(img_vector)

    # Cada fila es una imagen
    X = np.array(imagenes_vectorizadas)

    # Clase correspondiente a cada imagen
    y = etiquetas["clase"].to_numpy()

    return X, y

X_train, y_train = cargar_datos(carpeta_dataset, "train")
X_test, y_test = cargar_datos(carpeta_dataset, "test")

def regression_logistica(X_train, y_train, X_test, y_test):
    
    modelo = LogisticRegression(max_iter=2000) # Crear el modelo de regresión logística
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) # Calcular la accuracy del modelo

    return accuracy

accuracy_sin_pca = regression_logistica(X_train, y_train, X_test, y_test)
print("1-a) Accuracy sin PCA:", accuracy_sin_pca)

#-----------------------------EJERCICIO B----------------------------------------------#

def aplicar_pca(X_train, X_test, K):

    pca = PCA(n_components=K, random_state=42)
    
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca

K = 2
X_train_pca, X_test_pca, pca = aplicar_pca(X_train,X_test,K)

print("1-b) Antes de PCA:", X_test.shape)
print("     Después de PCA:", X_test_pca.shape)

def graficar_pca(X_pca, y, titulo):
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

graficar_pca(X_test_pca,y_test,"PCA - Conjunto de test")

#-----------------------------EJERCICIO C----------------------------------------------#

def evaluar_pca(X_train, y_train, X_test, y_test, valores_K):

    accuracies = []

    for K in valores_K:

        # Aplicar PCA con K componentes
        X_train_pca, X_test_pca, _ = aplicar_pca(X_train, X_test, K)

        # Entrenar y evaluar regresión logística
        accuracy = regression_logistica(X_train_pca, y_train, X_test_pca, y_test)

        accuracies.append(accuracy)

        print(f"K = {K}, Accuracy = {accuracy:.4f}")

    return accuracies

def graficar_accuracy_pca(valores_K, accuracies_pca, accuracy_sin_pca):

    plt.figure(figsize=(8, 6))

    # Accuracy usando PCA
    plt.plot(valores_K, accuracies_pca, marker="o", label="PCA + Regresión Logística")

    # Accuracy del punto 1(a)
    plt.axhline(y=accuracy_sin_pca,linestyle="--",label="Regresión Logística sin PCA")

    plt.xlabel("Número de componentes principales K")
    plt.ylabel("Accuracy")
    plt.title("Accuracy en función de K")
    plt.legend()
    plt.grid()

    plt.show()

print("1-c) Evaluando PCA con diferentes valores de K:")
valores_K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100, 150, 200]

accuracies_pca = evaluar_pca(X_train, y_train, X_test, y_test, valores_K)

graficar_accuracy_pca(valores_K, accuracies_pca, accuracy_sin_pca)