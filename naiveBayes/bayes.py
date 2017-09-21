from numpy import *


# 准备数据
def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        # else:
        #     print('the word: %s is not in my Vocabulary!' % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):  # 类别文档，总词条
    numTrainDocs = len(trainMatrix)  #
    numWords = len(trainMatrix[0])  #
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom, p1Denom = 2.0, 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    return 1 if p1 > p0 else 0


def testNB():
    listOPost, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPost)
    # print(myVocabList)
    # print(setOfWords2Vec(myVocabList, ['my']))

    p0V, p1V, pAb = trainNB0(listOPost, listClasses)
    # print('p0Vect:', p0V)
    # print('p1Vect:', p1V)
    # print('pAbusive:', pAb)

    trainMatrix = []
    for postinDoc in listOPost:
        trainMatrix.append(setOfWords2Vec(myVocabList, postinDoc))

    testEntry = ['my', 'love', 'dog']
    vec2Classify = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as ', classifyNB(vec2Classify, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    vec2Classify = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as ', classifyNB(vec2Classify, p0V, p1V, pAb))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList, classList, fullText = [], [], []
    for i in range(1, 26):
        print(i)
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 测试
    for i in range(10):  # 随机构建测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat, trainClasses = [], []
    # 训练
    for docIndex in trainingSet:  # 剩下训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pAb = trainNB0(trainMat, trainClasses)

    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pAb) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:%f' % float(errorCount / len(testSet)))


if __name__ == '__main__':
    spamTest()
