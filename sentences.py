# importing required modules
import PyPDF2
import os
import csv as csv
cwd = os.getcwd()# creating a pdf file object
# print("cwd is :")
# print(cwd)
sentences = []
def getsentences():
    for i in range(1,458):
        files = cwd + '/files/'+str(i)+'.pdf'
        pdfFileObj = open(files, 'rb')
        strin = ''
        # creating a pdf reader object
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        # printing number of pages in pdf file
        for j in range(pdfReader.numPages):

            # creating a page object
            strin += pdfReader.getPage(j).extractText()

            # extracting text from page
            # print(strin)

        sentences.append(strin)    # closing the pdf file object
        pdfFileObj.close()
    print "All pdf files are read!!"
    return sentences

def evalsetWriter():
    file = open('EvalSetNew.txt', 'a+')
    files = ''
    for i in range(1,458):
        files = str(i)+';'+ cwd + '/files/' + str(i) + '.pdf;'+str(i)+'\n'
        file.write(files)
    file.close()
    return 0


def evalnumWriter():
    file = open('eval_nums.txt', 'a+')
    files = ''
    for i in range(250,459):
        files = str(i)+'\n'
        file.write(files)
    file.close()
    return 0


def finalEval():
    file = open('finalEval_test1.txt', 'a+')

    f = open('Annotation.csv','r')
    reader = csv.reader(f)
    files = ''
    for row in reader:
        files = row[0]+';'+row[4]+';'+row[2]+';'+row[3]+'\n'
        file.write(files)
        files = ''
    file.close()
    return 0

if __name__ == '__main__':
    #getsentences()
    #evalsetWriter()
    #evalnumWriter()
    finalEval()