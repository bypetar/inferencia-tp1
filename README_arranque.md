# TP1 Inferencia y Estimacion

## Que hay en esta carpeta

- `TP1_IE (26b).pdf`: consigna del trabajo practico.
- `dataset_tp1/`: dataset de imagenes.
- `dataset_tp1/train/`: imagenes para entrenamiento.
- `dataset_tp1/test/`: imagenes para prueba.
- `dataset_tp1/train_labels.csv`: etiquetas de entrenamiento.
- `dataset_tp1/test_labels.csv`: etiquetas de prueba.

## Plan para empezar

1. Leer completa la consigna y separar lo que pide cada ejercicio.
2. Revisar el dataset: cantidad de imagenes, formato, tamanio y balance entre clases.
3. Empezar por el Ejercicio 1a: entrenar Logistic Regression con las imagenes completas, sin PCA, y guardar ese accuracy como referencia.
4. Seguir con el Ejercicio 1b: aplicar PCA con dos componentes principales y hacer el scatter del conjunto de test diferenciando las clases por color.
5. Resolver el Ejercicio 1c: repetir PCA + Logistic Regression para varios valores de K y graficar accuracy vs. K junto con el accuracy sin PCA.
6. Para el Ejercicio 2, entrenar una sola vez PCA + Logistic Regression con K = 2 usando train sin perturbar.
7. Aplicar Monte Carlo sobre test: en cada simulacion, rotar algunas imagenes 180 grados con probabilidad p.
8. Para cada p, estimar el accuracy medio, la probabilidad de que la perdida supere delta = 0.1 y hacer los histogramas de accuracies.
9. Resolver el Ejercicio 3 justificando el estimador de probabilidad mediante la Ley de los Grandes Numeros.
10. Escribir el informe final en PDF y preparar el ZIP de entrega sin incluir la carpeta de imagenes.

## Orden sugerido de trabajo

Primero conviene obtener el baseline sin PCA. Despues se analiza PCA con K variable. Recién al final se hace Monte Carlo, porque depende de tener entrenado y entendido el modelo PCA + Logistic Regression con K = 2.
