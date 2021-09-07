import plotly
import json
import pandas as pd
el_json = open("clasification_report.txt", encoding="utf-8")

metricas = json.load(el_json)
metrica = json.dumps(metricas, indent = 2)
print(metrica)
df = pd.DataFrame(columns=["categoria","precision","recall","f1-score","support","Categorias","Datos de entrenamiento"])

data = {'col_1': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], 'glosadep2': [41,7,23,0,140,17,2,14,8,172,160,9,1,0,50], "Datos de entrenamiento": [95, 29, 39,0,495,58,11,36,0,553,467,31,2,0,158]}
base_de_datos=pd.DataFrame.from_dict(data)

lista_de_categorias = ["publico, positivo","publico, neutral","publico, negativo","","publico, none","privado, positivo","privado, neutral","privado, negativo","","privado,none","lugar, positivo","lugar, neutral","lugar, negativo","","lugar, none"]

categorias_usadas = [0,1,2,4,5,6,7,9,10,11,12,14]

for x in categorias_usadas:
    try:

        a = metricas[str(x)]["precision"]
        b = metricas[str(x)]["recall"]
        c = metricas[str(x)]["f1-score"]
        d = metricas[str(x)]["support"]
        # lista = [a, b, c, d]
        lista = pd.DataFrame({"categoria" : [x]
                                ,"precision" :  [a]
                                ,"recall" : [b]
                                ,"f1-score" : [c]
                                ,"support" : lista_de_categorias[x]
                                ,"Categorias": 0
                                ,"Datos de entrenamiento": ""})
        df = df.append(lista)
        y = base_de_datos.at[x,"glosadep2"]
        #xplus = xplus + 1
        df.loc[df.categoria == x, 'Categorias'] = y+base_de_datos.at[x,"Datos de entrenamiento"]
        df.loc[df.categoria == x, 'Datos de entrenamiento'] =  base_de_datos.at[x,"Datos de entrenamiento"]
    except:
        pass

print(df)

fig = plotly.graph_objects.Figure(data=[plotly.graph_objects.Table(
                                header=dict(values=df.columns),
                                cells=dict(values=[df["categoria"],df["precision"],df["recall"],df["f1-score"],df["support"],df["Categorias"],df["Datos de entrenamiento"]],
               align='left'))
                     ])
fig.show()


import plotly.express as px
fig = px.scatter(df, 
                                x="Datos de entrenamiento", 
                                y="f1-score", 
                                size="f1-score",
                                color="support",
                                hover_name="recall", 
                                log_x=True, 
                                size_max=20)
fig.show()