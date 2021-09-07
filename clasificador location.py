#Cargar librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import json

#hiperparametros
muestra_de_labels = False
modelo_usado = "LSTM"
numero_de_labels = 3 # establecer cuantos labels se usan para testear modelo, todos serian 1026
embedding_dim = 300 # La dimensión de los embeddings es un parámetro que, a medida que cambia, genera diferentes resultados.
optimizador = "RMSprop" # Adam y Nadam son alternativas comunes.
loss = "sparse_categorical_crossentropy"
epochs = 10 # este parámetro determina la cantidad de veces que el modelo ve la base de entrenamiento. Normalmente de tres a cinco veces es suficiente en los modelos de NLP, muchos entrenamientos generan sobreajuste.
batch_size = 1000 # El tamaño del lote determina la cantidad de datos procesados al mismo tiempo, una gran cantidad genera que la computadora no sea capaz de procesarlo, muy pocos provocan que el entrenamiento del modelo sea muy lento, por lo que lo ideal es que este número sea el mayor posible sin que la computadora se bloquee.

base_clasificada = pd.read_csv("datos_clasificacion.csv", sep=";")
from sklearn.utils import shuffle
base_clasificada = shuffle(base_clasificada)
print(base_clasificada)
# base_clasificada = dbase_clasificada.drop_duplicates(subset='favorite_color', keep="first")
base_clasificada = base_clasificada[base_clasificada["en chile o no manual"] != 'location not categorized']
base_clasificada = base_clasificada[base_clasificada["en chile o no manual"] != 'i']


base_clasificada["en chile o no manual"] = base_clasificada["en chile o no manual"].replace({'s': "1", 'n': "0"})
print(base_clasificada["en chile o no manual"])

sentences = base_clasificada['ruta'].values
y = base_clasificada['en chile o no manual'].values

ruta_tweets = "/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/"
for idx, val in enumerate(sentences):
    tweet = open(ruta_tweets + val.replace("filename ", "") , encoding="utf-8")
    tweet_dict = json.load(tweet)
    texto_location_tweet = tweet_dict["user"]["location"]
    sentences[idx] = texto_location_tweet
print(sentences)

df = pd.DataFrame(list(zip(sentences, y)), 
               columns =['sentences', 'y'])
df = df.drop_duplicates(subset='sentences', keep="first")
 

sentences = df['sentences'].values
y = df['y'].values


sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.1, random_state = 1000, stratify = y)

print(sentences_train)
print(sentences_test)
print(y_train)
print(y_test)

y_train = y_train.astype(np.float)
y_test = y_test.astype(np.float)

# Tokenizar texto
tokenizer = Tokenizer(num_words=1000,  oov_token = "OOV") # Aquí pones el número de tokens junto al token que recibe las palabras fuera del vocabulario. Una cantidad demasiado grande de tokens puede hacer que se manejen mal palabras poco comunes.
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1

print(sentences_train[1])
print(X_train[1])

# Generar el padding. Se pone al principio las palabras y luego se completa con ceros
maxlen = 50
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(X_train[2, :])


# Definir arquitectura de la red y compilar el modelo

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Bidirectional(layers.LSTM(embedding_dim)))
model.add(layers.Dense(embedding_dim, activation='relu'))
model.add(layers.Dense(numero_de_labels, activation='softmax'))


model.compile(optimizer=optimizador,
              loss=loss,
              metrics=['accuracy'])
model.summary()


# Entrenar el modelo 
history = model.fit(X_train, y_train,
                    epochs = epochs,
                    verbose=False,
                    validation_split = 0.1,
                    #validation_data=(X_test, y_test),
                    batch_size = batch_size)
                    
                    
# Train accuracy
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
                    
print(history.history['val_accuracy'])
print(history.history['accuracy'])


# Testear en datos que la red aún no ha visto
print("test accuracy")
predict = model.predict_classes(X_test, verbose=0)

# Ordenar tabla de salida con las predicciones
pred_df = pd.DataFrame(predict, columns=['predicho']) 
y_test_df = pd.DataFrame(y_test, columns=['real']) 
#glosas = pd.DataFrame(sentences_test, columns=['glosa_original'])
pred_df2 =  pd.concat([pred_df, y_test_df], axis=1, sort=False)

# Mirar porcentaje de coincidencia
comparar = np.where(pred_df2["real"] == pred_df2["predicho"], True, False)
print("performance test data set")
print(comparar.mean())


                    
# train and validation loss
import matplotlib.pyplot as plt
plt.style.use('ggplot')

numero_de_embeddings = str(embedding_dim)
numero_de_labels_string = str(numero_de_labels)
performance_test = str(comparar.mean())
nombre_prueba = "clasificador de ubicacion, Modelo " + modelo_usado + ", performance test dataset " + performance_test + ", optimizador " + optimizador + ", labels " + numero_de_labels_string # Aquí se nombra el JSON que entrega los resultados del modelo.


x = sklearn.metrics.classification_report(y_test, predict, output_dict=True)

import json

j = open("loss_models/" + nombre_prueba + ".txt", "w", encoding="utf-8")
j.write(json.dumps(x, ensure_ascii=False))
j.close()


model.save("models/lstm, performance modelo " + performance_test + ".h5")  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

import pickle

# saving
with open("tokenizer/token, performance model"+ performance_test +".pickle", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


from tensorflow.keras.models import load_model
# import pickle
model = load_model("models/lstm, performance modelo " + performance_test + ".h5")

print(type(model))


# loading
with open("tokenizer/token, performance model"+ performance_test +".pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)


# import numpy as np
ejemplo_de_frase = tokenizer.texts_to_sequences(["soy de mexico"])

print(ejemplo_de_frase)

# Generate predictions for samples
predictions = model.predict(ejemplo_de_frase)
print(predictions)

# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print(classes)