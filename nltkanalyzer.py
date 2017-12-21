import unirest as unirest

from sentences import getsentences # sentences.py reads all the pdf files in as List of single sentences

sentencesList = []  # holds all sentences read from pdf files
flag = 0
def setdebug(flag):
    flag = flag

def analyze(sentence):
    # TODO: use NLTK sentiment analyzer --> Done
    response = unirest.post("https://japerk-text-processing.p.mashape.com/sentiment/",
                            headers={
                                "X-Mashape-Key": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                                "Content-Type": "application/x-www-form-urlencoded",
                                "Accept": "application/json"
                            },
                            params={
                                "language": "english",
                                "text": sentence
                            }
                            )
    return response


def printoutput(start, end):
    # TODO: Make bad list --> Done
    sentencesList = getsentences()
    badList = open('badlist_nltk.txt', 'a+') # contains list of all the bad files negative code files
    goodList = open('goodlist_nltk.txt', 'a+') # contains list of all the good files
    for i in range(start,end):
        response = analyze(sentencesList[i]) # get response from API
        if flag == 1 :
            print ["for file ", i, " : ", response.body['label'], response.body['probability']]
        if response.body['label'] == 'neg': # response is negative then push into bad list
            badList.write(str(i)+'\n')
        else:                                # response is positive then push into good list
            goodList.write(str(i)+'\n')
    print "Done geerating good or bad List"

if __name__ == '__main__':
    setdebug(1)
    printoutput(251,458) # for Me
    # printoutput(1,250) # for Abdullah and Eeshita
