import pandas as pd 

df = pd.read_csv("datos_clasificacion.csv" ,sep=";")

df = df[df["target manual"] != "[]"]

print(df.groupby('en chile o no manual').count())