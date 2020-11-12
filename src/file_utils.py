import re


def readFile(evaluationfile):
    tuplas=[]
    puntos=[]
    with open(evaluationfile, 'r') as f:
        for line in f: #TODO extender a mas tuplas
            #regexp= re.compile(r'(.+)@(.+)@(.+) (\d*\.\d*)\n')
            regexp = re.compile(r'(.+@.+) (.+@.+) (\d*\.\d*)\n')
            if re.match(regexp,line):
                datos=re.search(regexp,line).groups()
                tuplas.append((datos[0],datos[1]))
                puntos.append(float(datos[2]))
    return tuplas,puntos
