import csv
from prettytable import PrettyTable
file = open('configurations.csv', 'r')
reader = csv.reader(file)
t = PrettyTable(['Configuration', 'Precision', 'Recall','True positive','True negative','False positive','False negative'])
prec = 0
rec = 0
i = 0
for row in reader:
    t.add_row([row[0],row[2], row[1],row[3],row[4],row[5],row[6]])
    if i <13 :
        prec += float(row[2])
        rec += float(row[1])
    i = i+1

print "Average precision : " + str(prec/13)
print "Average recall : " + str(rec/13)

print t