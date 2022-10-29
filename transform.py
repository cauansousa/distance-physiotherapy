import json
import math
import os
import glob

all_angles = []

#ler todos os arquivos json
for file in glob.glob("data/*.json"):
    f = open(file)
    data = json.load(f)
    f.close()
    
    #para cada arquivo json, ler todos os frames
    #para cada frame, ler todas as juntas
    all_angles.append(data)
with open('all_angles.json', 'w') as outfile:
    json.dump(all_angles, outfile)