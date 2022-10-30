import json
import math
import os
import glob

all_angles = []

#ler todos os arquivos json
for file in glob.glob("*.json"):
    f = open(file)
    data = json.load(f)
    f.close()
    
    #para cada arquivo json, ler todos os frames
    #para cada frame, ler todas as juntas
    all_angles.append(data)
print(all_angles)
with open('all_angles.json', 'w') as outfile:
    json.dump(all_angles, outfile)