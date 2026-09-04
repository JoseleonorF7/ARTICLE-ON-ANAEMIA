# ARTICLE ON ANAEMIA

Repositorio con los códigos utilizados para realizar los experimentos del artículo sobre detección no invasiva de anemia.

## Códigos

### `PREPROCESAMIENTO DATASET ANEMIA.ipynb`

Este cuaderno se utilizó **localmente** para procesar los videos del conjunto de datos de **San Martín**.

Su función principal es transformar los videos en imágenes procesadas de la palma de la mano que posteriormente se utilizaron en los experimentos.

**Uso:**

1. Obtener los videos del conjunto de datos.
2. Abrir el archivo en Jupyter Notebook.
3. Revisar y modificar las rutas de entrada y salida.
4. Ejecutar las celdas para realizar el preprocesamiento.

---

### `MODELS- ANEMIA.py`

Este es el **código principal del artículo** y fue ejecutado en **Kaggle**.

Contiene el código utilizado para realizar el entrenamiento, validación y evaluación de los modelos, así como los análisis realizados en el estudio, incluyendo **Grad-CAM**.

**Uso:**

1. Obtener los conjuntos de datos utilizados en el estudio.
2. Configurar las rutas de los datos en el código.
3. Ejecutar el script en un entorno compatible con Python/PyTorch, como Kaggle.
4. Los resultados generados corresponden a los experimentos descritos en el artículo.

## Datos

Los conjuntos de datos **no están incluidos en este repositorio**. Para utilizarlos, deben obtenerse desde las fuentes correspondientes indicadas en el artículo y respetar sus condiciones de uso.

## Repositorio

[GitHub — ARTICLE-ON-ANAEMIA](https://github.com/JoseleonorF7/ARTICLE-ON-ANAEMIA?utm_source=chatgpt.com)

Este repositorio se proporciona para facilitar la consulta, transparencia y reproducción de los procedimientos computacionales utilizados en el estudio.
