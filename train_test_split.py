#Cargar librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# usar parte de la base de datos respetando las categorias
ruta = "/Users/joaqu/OneDrive/Documentos/drive/twitter/"
path = "datos_clasificacion"
df = pd.read_csv(path + ".csv", encoding='utf-8', usecols=col_list, engine="python", error_bad_lines=False)


# extraer categorias que no cumplan un minimo de 2 casos
minimo_casos_categoria = 2 # Esto tiene que hacerse porque, en la función que divide la base de datos en entrenamiento y prueba, existe el argumento de que para funcionar eso exige más de un caso por categoría.
ccif_df = ccif_df.groupby('codigo2').filter(lambda x: len(x) > minimo_casos_categoria)
print(ccif_df.shape)
print(ccif_df.head(100))
print(len(ccif_df.codigo2.value_counts())) # Agrega la columna con etiquetas.
# print(ccif_df.to_string()) # permite ver el contenido del df completo

# generar una muestra mas pequeña para entrenar
import random

# Separar train y test. El test set solo se mira al final 
sentences = ccif_df['glosadep2'].values
y = ccif_df['codigo2'].values # Agrega la columna con etiquetas.
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.1, random_state = 1000, stratify = y) # Aquí eliges el porcentaje que quieres que tenga la base de prueba, generalmente se divide en un ratio de 80/20, pero tienes que elegir lo que es apropiado en tu caso particular. El argumento final permite dividir la base de datos de manera que se respete la proporcionalidad de las categorías en la base de datos original. Por ejemplo, si se extrae el 10% de una base de datos para realizar pruebas, una categoría con 100 casos tendrá 10 en la base de pruebas. Por tanto, se asegura que las categorías más pequeñas tengan representación en ambas bases de datos.

print(y_train.shape)
print(pd.Series(y_train).value_counts())

msk = np.random.rand(len(ccif_df)) < 0.9
train = ccif_df[msk]
test = ccif_df[~msk]
print(len(test))
print(len(train))
print(train.head())

train.to_csv(path + "_train.csv", encoding='iso-8859-1')
test.to_csv(path + "_test.csv", encoding='iso-8859-1') # Asegúrese de que el método de codificación sea el adecuado.