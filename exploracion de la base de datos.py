import pandas as pd

df = pd.read_csv("datos_clasificacion.csv",sep=";")

print(df["trata la sequia"].unique()) 