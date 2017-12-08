import matplotlib.pyplot as plt
import csv
import pylab
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import numpy as np


def plotgraph(values,ylabel, xlabel,xvalues,name):
    #plt.plot(xvalues,values[0],'b-', xvalues, values[1],'r-')
    pylab.plot(xvalues,values[0],'bo',label='precision')
    pylab.plot(xvalues,values[0],'k')
    pylab.plot(xvalues,values[1],'ro',label='recall')
    pylab.plot(xvalues,values[1],'k')
    pylab.xlabel(xlabel)
    pylab.ylabel("Precision and Recall")
    #plt.ylabel(ylabel)
    #plt.xlabel(xlabel)
    pylab.legend(loc='upper left')
    # pylab.show()
    fig1=pylab.gcf()
    pylab.draw()
    # plt.axis([0,13,0.25,0.4])
    # plt.figure()
    # plt.show()
    #pdffig = PdfPages(name)
    #fig1 = plt.gcf()
    #plt.draw()
    fig1.savefig(name, format="pdf")
    '''d = fig1.infodict()
    d['Title'] = 'Plots with All configuration'
    d['Author'] = u'Aditya Karmarkar'
    d['Subject'] = 'Sentiment Analysis'
    d['Keywords'] = 'Sentiment Analysis'
    d['CreationDate'] = datetime.datetime(2017, 12, 02)
    d['ModDate'] = datetime.datetime.today()
    '''

def getValues(decision,reader,yvalues):
    for row in reader:
        if decision == "precision" :
            yvalues.append(row[2])
        if decision == "recall":
            yvalues.append(row[1])
    return yvalues

def addinfo(pdf):
    d = pdf.infodict()
    d['Title'] = 'Plots with All configuration'
    d['Author'] = u'Aditya Karmarkar'
    d['Subject'] = 'Sentiment Analysis'
    d['Keywords'] = 'Sentiment Analysis'
    d['CreationDate'] = datetime.datetime(2017, 12, 02)
    d['ModDate'] = datetime.datetime.today()


def barcharts(xvalues,values):
    n_groups = len(xvalues)

    means_men = values[0]
    #std_men = (2, 3, 4, 1, 2)

    means_women = values[1]
    # std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.45

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, means_men, bar_width,
                     alpha=opacity,
                     color='b',
                     # yerr=std_men,
                     error_kw=error_config,
                     label='Precision')

    rects2 = plt.bar(index + bar_width, means_women, bar_width,
                     alpha=opacity,
                     color='r',
                     # yerr=std_women,
                     error_kw=error_config,
                     label='Recall')

    plt.xlabel('Configurations')
    plt.ylabel('Scores')
    plt.title('Scores by Configuration')
    plt.xticks(index + bar_width/2 , xvalues)#('Conf 1', 'Conf 2', 'Conf 3', 'Conf 4', 'Conf 5', 'Conf 6', 'Conf 7', 'Conf 8', 'Conf 9', 'Conf 10','Conf 11','Conf 12', 'Conf 13'))
    plt.legend()

    plt.tight_layout()
    fig1=plt.gcf()
    pylab.draw()
    fig1.savefig('barcharts.pdf',format="pdf")

    plt.show()


yvalues = []
# xvalues = ["bad_5Weight100bi_0uni", "bad_5Weight100bi_20uni", "bad_10Weight100bi_0uni", "bad_10Weight100bi_20uni",
#           "bad_100Weight100bi_20uni", "bad_100Weight25bi_0uni", "bad_100Weight25bi_20uni",
#           "bad_100Weight50bi_0uni", "bad_100Weight50bi_20uni", "bad_100Weight75bi_0uni", "bad_100Weight75bi_20uni",
#           "bad_100Weight100bi_0uni", "bad_100Weight100bi_20uni"]
xvalues = [1,2,3,4,5,6,7,8,9,10,11,12,13]
file = open('configurations.csv', 'r')
reader = csv.reader(file)
print "plot1"
'''
name = 'precision.pdf'
plot1 = plotgraph(getValues("precision",reader,yvalues),"precision","configurations",xvalues,name)
print "plot2"
name = 'recall.pdf'
plot2 = plotgraph(getValues("recall",reader,yvalues),"recall", "configurations",xvalues,name)
print "saving"
'''
yvalues = getValues("precision",reader,yvalues)
name = "final.pdf"
y2values = []
file.close()
# reader.close()
file = open('configurations.csv', 'r')
reader = csv.reader(file)
y2values = getValues("recall",reader,y2values)
#plotgraph([yvalues,y2values], "","configurations",xvalues,name)
#file.close()
barcharts(xvalues,[yvalues,y2values])
# reader.close()
# pp.savefig(plot3)