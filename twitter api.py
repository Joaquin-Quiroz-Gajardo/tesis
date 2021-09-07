import tweepy
import os 
import datetime

date = datetime.datetime.now()
date = str(date.strftime("%d")) + " " + str(date.strftime("%b"))
os.mkdir("/content/drive/MyDrive/twitter/tweets final/" + date)

consumer_key = ""
consumer_secret = ""

access_token = ""
access_token_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

numero = 0
import json
for tweet in tweepy.Cursor(api.search , q = "sequía", tweet_mode = "extended").items(5000):
    numero += 1
    numero_2 = str(numero)
    nombre = "/content/drive/MyDrive/twitter/tweets final/"+ date + "/" + "tweet numero "+ numero_2 + ".txt"
    j = open(nombre, "w", encoding="utf-8")
    j.write(json.dumps(tweet._json, ensure_ascii=False))
    j.close()