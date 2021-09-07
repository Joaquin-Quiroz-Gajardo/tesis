import json
import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

# loading location0.7916666666666666
import pickle
model = load_model('models/lstm, performance modelo 0.7708333333333334.h5')

print(type(model))

with open('tokenizer/token, performance model0.7708333333333334.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# loading trata sequia 0.7307692307692307
model_trata_sequia = load_model('models/lstm trata sequia o no, performance modelo 0.8347826086956521.h5')

print(type(model))

with open('tokenizer/token trata sequia o no, performance model0.8347826086956521.pickle', 'rb') as handle:
    tokenizer_trata_sequia = pickle.load(handle)


base_clasificada = pd.read_csv("datos_clasificacion.csv", sep=";", encoding="utf-8")

base_clasificada["trata la sequia"] = base_clasificada["trata la sequia"].replace({'s': "1", 'n': "0"})
print(base_clasificada["trata la sequia"])

sentences = base_clasificada['ruta'].values
y = base_clasificada['trata la sequia'].values

ruta_tweets = "/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/"
for idx, val in enumerate(sentences):
    tweet = open(ruta_tweets + val.replace("filename ", "") , encoding="utf-8")
    tweet_dict = json.load(tweet)
    texto_location_tweet = tweet_dict["full_text"]
    sentences[idx] = texto_location_tweet
# print(text)
print(sentences)


df = pd.DataFrame(list(zip(sentences, y)), 
               columns =['sentences', 'y'])
df = df.drop_duplicates(subset='sentences', keep="first")
 

sentences = df['sentences'].values
y = df['y'].values

already_reviewed = list()

from nltk.corpus import stopwords
from collections import Counter
import re
# text = "le dije hola a los perros corren por la pradera"
stop_words = stopwords.words('spanish')
# stop_words.extend(["hola","chao"])
# print(stop_words)
stopwords_dict = Counter(stop_words)


from glob import glob
import os
all_directories = glob("C:/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/*/")
for directory in all_directories:
    dictionary_to_json = dict()
    directory = directory.replace("\\", "/")
    ya_fue_clasificada = False
    for filename in os.listdir(directory):
        if filename == "clasification.txt":
            ya_fue_clasificada = True
        # if filename.endswith(".txt"):
    if directory == "C:/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/loss_models/":
        pass
    elif ya_fue_clasificada == True:
        print("ya ha sido clasificado")
    else:
        print(directory)

        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                print(os.path.join(directory, filename))
                with open(directory + filename , encoding="utf-8") as json_data:
                    
                    try:
                        d = json.load(json_data)
                    except:
                        print("JSONDecodeError")
                        continue
                    
                    lista = []
                    ubicacion_declarada = d["user"]["location"]
                    
                    if ubicacion_declarada == "" or ubicacion_declarada == " " or ubicacion_declarada == "  " or ubicacion_declarada == "   " or ubicacion_declarada == "." or  ubicacion_declarada ==  "     " or ubicacion_declarada == ";)" or ubicacion_declarada == "??" or ubicacion_declarada == "                             ;":
                        ubicacion_declarada = "argentina"

                    

                    lista.append(filename)
                    x = d["full_text"]
                    if x[:2] == "RT":
                        continue
                    
                    # import numpy as np
                    try:
                        ubicacion_tokens = tokenizer.texts_to_sequences([ubicacion_declarada])
                    except:
                        print("location rara")
                        continue
                    # print(ubicacion_tokens)

                    # Generate predictions for samples
                    try:
                        predictions = model.predict(ubicacion_tokens)
                    except:
                        continue
                    # print(predictions)

                    # Generate arg maxes for predictions
                    classes = np.argmax(predictions, axis = 1)


                    if classes == [0]:
                        
                        

                        print("no chile")
                        continue
                    
                    # modelo de sequia
                    x = re.sub("@[a-zA-Z0-9áéíóúñ_-]*", "hastag", x)
                    x = re.sub("#[a-zA-Z0-9áéíóúñ_-]*", "user", x)
                    x = re.sub("https:\/\/[a-zA-Z0-9áéíóúñ.\/]*", "link", x)
                    x = ' '.join([word for word in x.split() if word not in stopwords_dict])
                    trata_sequia_tokens = tokenizer_trata_sequia.texts_to_sequences([x])

                    # print(ubicacion_tokens)

                    # Generate predictions for samples
                    predictions = model_trata_sequia.predict(trata_sequia_tokens)
                    # print(predictions)

                    # Generate arg maxes for predictions
                    trata_sequia = np.argmax(predictions, axis = 1)

                    if trata_sequia == [0]:
                        print("no trata la sequia")
                        continue

                    elif d["full_text"] in sentences or d["full_text"] in already_reviewed:
                        print("retweet manual")
                        continue
                    else:
                        already_reviewed.extend(d["full_text"])
                        print("→ " + d["full_text"])
                        print("¿trata la sequia? Si (s) o No (n)")
                        categoria = input()
                        lista.append(categoria)
                        print("Targets: Organismos públicos (p), Empresas privadas (e), Lugares (l) o ninguno mas (x)\ny su categoria: Positivo (p), Negativo (n), Neutral o Ecuanime (e) o Indeterminado (i)")
                        informacion = ""
                        if (lista[1] != "n" and lista[1] != ""):
                            while informacion != "x":
                                lista.append(informacion)
                                try:
                                    lista.remove('')
                                except ValueError:
                                    pass
                                informacion=input()
                            print(d["user"]["location"])
                            if(d["user"]["location"] == ""):
                                location = "i"
                                print("location is indeterminable")
                            else:
                                print("La ubicacion esta dentro de Chile, si (s), no (n) o es indeterminable (i)")
                                location = input()
                            lista.insert(2, location)
                        try:
                            lista.remove('x')
                        except ValueError:
                            pass
                        print(lista)
                        if len(lista) == 2:
                            lista.append("location not categorized")
                        if lista != []:
                            dictionary_to_json["filename " + lista[0]] = {"location":lista[2]
                                                                        , "trata la sequia":lista[1]}
                        number_range = len(lista)- 3
                        for i in range(number_range):
                            i = i+3
                            name_key = "target number " + str(i-3)
                            dictionary_to_json["filename " + lista[0]][name_key] = lista[i]
                            
                        j = open(directory + "clasification.txt", "w", encoding="utf-8")
                        j.write(json.dumps(dictionary_to_json, ensure_ascii=False))
                        j.close()