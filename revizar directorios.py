from glob import glob
import os
all_directories = glob("C:/Users/joaqu/OneDrive/Documentos/drive/twitter/tweets final/*/")
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
        print("ya ha sido clasificado")
    else:
        print(directory)