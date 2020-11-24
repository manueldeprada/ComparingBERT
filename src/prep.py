import sys
from pyskiplist import SkipList

sl = SkipList()

for file in sys.argv[1:]:
    with open(file, encoding="ansi") as infile:
        count = 1
        for line in infile:
            # if (line[0] != '<' and line[0] != '.' and line[0] != ','and line[0] != '"'and line[0] != ';'):
            if (line[0].isalpha() and line[0].isascii()):
                i = 0
                while len(line)>i and line[i] != '\t':
                    i += 1
                if len(line)>i+1 and not line[i + 1].isupper():
                    j = i + 1
                    save=True
                    while line[j]!='\t':
                        if (not line[j].isalpha() ) or not line[j].isascii():#and line[j]!='-'
                            save=False
                        j += 1
                    if save:
                        sl.replace(line[i + 1:j], None)
            if count % 1000000 == 0:
                print(str(count/271305487)+" "+str(count)+ " "+ str(sl.__len__()))
            count += 1

with open('dict_4.txt','w') as file:
    for i in sl.keys():
        file.write(i)
        file.write('\n')



