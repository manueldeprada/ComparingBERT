import sys

with open(sys.argv[1],encoding="ansi") as infile:
    for line in infile:
        if (line[0]!='<' and line != '.'):
            i=0
            while line[i]!='\t':
                i+=1
            j=i+1
            while line[j]!='\t':
                j+=1
            print(line[i:j])

