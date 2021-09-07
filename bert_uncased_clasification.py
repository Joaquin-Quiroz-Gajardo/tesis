import tensorflow as tf
# import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import tensorflow.keras
from tqdm import tqdm
import pickle
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import itertools
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig
import json


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w


base_clasificada = pd.read_csv("datos_clasificacion.csv", sep=";", encoding="utf-8")
from sklearn.utils import shuffle
base_clasificada = shuffle(base_clasificada)
print(base_clasificada)

base_clasificada["trata la sequia"] = base_clasificada["trata la sequia"].replace({'s': "1", 'n': "0"})
print(base_clasificada["trata la sequia"])

sentences = base_clasificada['ruta'].values
y = base_clasificada['trata la sequia'].values

from nltk.corpus import stopwords
from collections import Counter
import re
# text = "le dije hola a los perros corren por la pradera"
stop_words = stopwords.words('spanish')
# stop_words.extend(["hola","chao"])
# print(stop_words)
stopwords_dict = Counter(stop_words)

ruta_tweets = "/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/"
for idx, val in enumerate(sentences):
    tweet = open(ruta_tweets + val.replace("filename ", "") , encoding="utf-8")
    tweet_dict = json.load(tweet)
    texto_location_tweet = tweet_dict["full_text"]
    sentences[idx] = texto_location_tweet
    sentences[idx] = re.sub("@[a-zA-Z0-9áéíóúñ_-]*", "hastag", sentences[idx])
    sentences[idx] = re.sub("#[a-zA-Z0-9áéíóúñ_-]*", "user", sentences[idx])
    sentences[idx] = re.sub("https:\/\/[a-zA-Z0-9áéíóúñ.\/]*", "link", sentences[idx])
    sentences[idx] = ' '.join([word for word in sentences[idx].split() if word not in stopwords_dict])
# print(text)
print(sentences)

df = pd.DataFrame(list(zip(sentences, y)), 
               columns =['text', 'label'])
df = df.drop_duplicates(subset='text', keep="first")


df=df.dropna()# Drop NaN valuues, if any
df=df.reset_index(drop=True)# Reset index after dropping the columns/rows with NaN values
df = shuffle(df)# Shuffle the dataset
print('Available labels: ',df.label.unique())# Print all the unique labels in the dataset
df['text']=df['text'].map(preprocess_sentence)# Clean the text column using preprocess_sentence function defined above
print('File has {} rows and {} columns'.format(df.shape[0],df.shape[1]))

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
saver = tf.compat.v1.train.Saver()
# do some work with the model.
with tf.compat.v1.Session() as sess:
     # Restore variables from disk.
     saver.restore(sess, "/tensorflow/model.ckpt")
     print("Model restored.")
     # Do some work with the model