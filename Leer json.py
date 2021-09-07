import json
import glob
import tweepy

json_abrir = open("C:/Users/joaqu/OneDrive/Documentos/twitter/tweets final/01 May/tweet numero 1.txt"
, mode = "r"
, encoding = "utf-8")
el_json = json.load(json_abrir)

json_abrir.close()

def getList(dict): 
    return dict.keys() 

for tweet_iterado in glob.iglob("C:/Users/joaqu/OneDrive/Documentos/twitter/tweets final/19 Apr/*.txt"):
    json_en_loop = open(tweet_iterado
    , mode = "r"
    , encoding = "utf-8")
    json_en_loop = json.load(json_en_loop)
    print(json_en_loop["user"]["location"])
    print("------------------------------------------")