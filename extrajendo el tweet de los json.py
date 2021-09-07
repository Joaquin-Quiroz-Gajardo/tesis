import json

i = 0
nuevo_archivo = open("solo los tweets 2.txt","w",encoding="utf-8")

while (i < 10) :
    i+=1
    cargar_json_file = open("tweets/tweet numero " + str(i) +".txt","r",encoding="utf-8")
    json_tweet = json.load(cargar_json_file)
    cargar_json_file.close
    el_tweet = str(json_tweet["full_text"] + '\n')
    el_tweet.replace('\n', '. ')
    nuevo_archivo.write(el_tweet)

