import feedparser

from naiveBayes.bayes import *


def calMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList, classList, fullText = [], [], []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Word = calMostFreq(vocabList, fullText)

    for pairW in top30Word:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if (classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]):
            errorCount += 1

    print('the error rate is:%f' % float(errorCount / len(testSet)))
    return vocabList, p0V, p1V


if __name__ == '__main__':
    import feedparser

    ny = feedparser.parse('https://newyork.craigslist.org/search/stp?format=rss')
    sf = feedparser.parse('https://sfbay.craigslist.org/search/stp?format=rss')
    localWords(ny, sf)
    localWords(ny, sf)
    localWords(ny, sf)
    localWords(ny, sf)
