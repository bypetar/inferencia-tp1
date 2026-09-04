from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

carpeta_train = Path("dataset_tp1/train")

etiquetas_train = pd.read_csv("dataset_tp1/train_labels.csv")

imagenes_vectorizadas = []

for archivo in etiquetas_train["archivo"]:

    ruta_imagen = carpeta_train / archivo

    img = Image.open(ruta_imagen).convert("L")
    img_array = np.array(img)
    img_vector = img_array.reshape(-1)

    imagenes_vectorizadas.append(img_vector)
