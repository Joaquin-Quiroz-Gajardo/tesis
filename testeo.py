from lxml import etree
def find_str(s, char):
    index = 0

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index+len(char)] == char:
                    return index

            index += 1

    return -1

# abriendo la base de datos
import pandas as pd
base_clasificada = pd.read_csv("C:/Users/joaqu/OneDrive/Documentos/drive/twitter/datos_clasificacion.csv", encoding="UTF-8", error_bad_lines=False, sep=";")
from sklearn.utils import shuffle
base_clasificada = shuffle(base_clasificada)
print(base_clasificada)
base_clasificada = base_clasificada[base_clasificada["en chile o no manual"] != 'location not categorized']
base_clasificada = base_clasificada[base_clasificada["en chile o no manual"] != 'i']


# # transform string representation of list into list
import ast
# for index, lista_de_targets in enumerate(base_clasificada["target manual"]):
#     x = ast.literal_eval(lista_de_targets)
#     x = [n.strip() for n in x]


#crear nuevo dataframe donde esten los target y las sentences
sentences = base_clasificada['ruta'].values
target_list = base_clasificada['target manual'].values

# importando las sentencias
import json
ruta_tweets = "/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/"
for idx, val in enumerate(sentences):
    tweet = open(ruta_tweets + val.replace("filename ", "") , encoding="utf-8")
    tweet_dict = json.load(tweet)
    texto_location_tweet = tweet_dict["full_text"]
    sentences[idx] = texto_location_tweet
print(sentences)

df_to_create_xml = pd.DataFrame(list(zip(sentences, target_list)), 
               columns =['sentences', 'target_list'])
df_to_create_xml = df_to_create_xml.drop_duplicates(subset='sentences', keep="first")
import unicodedata
df_to_create_xml['sentences'] = df_to_create_xml['sentences'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())
df_to_create_xml['sentences'] = df_to_create_xml['sentences'].str.lower()

print(df_to_create_xml['sentences'].head(50))