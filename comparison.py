import matplotlib.pyplot as plt
import pylab
import numpy as np


def barchartsForComparison(xvalues,values,labels,names):
    n_groups = len(xvalues)

    ourtool = values[0]

    nltk = values[1]

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.4

    opacity = 0.8
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, ourtool, bar_width,
                     alpha=opacity,
                     color='b',
                     # yerr=std_men,
                     error_kw=error_config,
                     label=labels[0]) # our tool

    rects2 = plt.bar(index + bar_width, nltk, bar_width,
                     alpha=opacity,
                     color='r',
                     # yerr=std_women,
                     error_kw=error_config,
                     label=labels[1]) #python nltk

    plt.xlabel('Configurations')
    plt.ylabel('Scores')
    plt.title('Our tool vs Python Sentiment Analyzer')
    plt.xticks(index + bar_width/2 , xvalues)#('Conf 1', 'Conf 2', 'Conf 3', 'Conf 4', 'Conf 5', 'Conf 6', 'Conf 7', 'Conf 8', 'Conf 9', 'Conf 10','Conf 11','Conf 12', 'Conf 13'))
    plt.legend()

    plt.tight_layout()
    fig1=plt.gcf()
    pylab.draw()
    fig1.savefig(names,format="pdf")

    plt.show()


yvalues = [0.5,0.3076923076923077]
y2values = [0.2,0.8]
barchartsForComparison(['precision', 'recall'], [yvalues,y2values], ['Our Tool', 'NLTK Sentiment Analyzer'],'comparison.pdf')