import json
import pandas as pd
from glob import glob
import os
all_directories = glob("C:/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/*/")

df = pd.DataFrame({"ruta": []
                            ,"en chile o no machine": []
                            ,"en chile o no manual": []
                            ,"target machine": []
                            ,"target manual": []
                            ,"trata la sequia": []}) ###

for directory in all_directories:
    directory = directory.replace("\\", "/")
    ya_fue_clasificada = False
    for filename in os.listdir(directory):
        if filename == "clasification.txt":
            ya_fue_clasificada = True
        # if filename.endswith(".txt"):
    if directory == "C:/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/loss_models/":
        pass
    elif ya_fue_clasificada == True:
        print(directory)
        json_data = open(directory + "clasification.txt" , encoding="utf-8")
        d = json.load(json_data)

        date = directory.replace("C:/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final", "")
        # print(d["filename tweet numero 987.txt"])


        for x in d:
                number_of_keys = len(x)

                trata_sequia = d[x]["trata la sequia"]
                
                if trata_sequia == "" or trata_sequia == "m" or trata_sequia == "b" or trata_sequia == " ":
                    trata_sequia = "n"
                
                if len(trata_sequia) > 1 or trata_sequia == "a" or trata_sequia == "d" or  trata_sequia == "S":
                    trata_sequia = "s"
                tweet = open(directory + x.replace("filename ", "") , encoding="utf-8")
                tweet_dict = json.load(tweet)
                la_lista = d[x].keys()
                    
                try:
                    targets = list(la_lista)[2:]
                except:
                    targets = []

                # print(tweet_dict["user"]["location"] + " | " + tweet_dict["full_text"])
                # try:
                #     trata_la_sequia = d[x]["trata la sequia"]
                # except KeyError:
                #     trata_la_sequia = "s"
                list_of_targets = list()

                # print("d x",d[x])
                for index_t,value_t in enumerate(targets):
                    # print("index",index_t,"value",value_t,"x",d[x])
                    if len(value_t) <=2:
                        pass
                    try:
                        list_of_targets.append(d[x][value_t])
                        print("funciono", list_of_targets)
                    except:
                        print("no funciono",targets)
                targets=list_of_targets


                df2 = pd.DataFrame.from_dict({"ruta": [date + x]
                                    ,"en chile o no machine": [""]
                                    ,"en chile o no manual": [d[x]["location"]]
                                    ,"target machine": [""]
                                    ,"target manual": [str(targets)]
                                    ,"trata la sequia": [trata_sequia]})
                # print(df2.head())
                df = df.append(df2)

        print(df)
    else:
        print(directory)

import unicodedata

df['target manual'] = df['target manual'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())

df.to_csv("datos_clasificacion.csv", sep=";", index=False, encoding="ascii")
