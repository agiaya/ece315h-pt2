oldfile = open("training.csv", 'r')
newfile = open("training_exp.csv", 'w')

count = 11 * [0]
max = [1884,3373,649,365,172,108,635,106,840,304,2073]
#max = [800,800,800,440,200,130,800,130,600,360,800]

first = True
for line in oldfile:
    if first:
        newfile.write(line)
        first = False
    else:
        l = line.split(',')
        i = int(l[6])
        if count[i] < max[i]:
            newfile.write(line)
            count[i] += 1

oldfile.close()
newfile.close()