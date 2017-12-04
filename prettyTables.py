import csv
from prettytable import PrettyTable
file = open('configurations.csv', 'r')
reader = csv.reader(file)
t = PrettyTable(['Configuration', 'Precision', 'Recall','True positive','True negative','False positive','False negative'])
for row in reader:
    t.add_row([row[0],row[2], row[1],row[3],row[4],row[5],row[6]])
print t