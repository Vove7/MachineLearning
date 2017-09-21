import numpy as np
import os
import kNN


def img2Vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(line_str[j])

    return returnVect


# 获取labels
def getMatAndLabels():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileName = trainingFileList[i]
        classNum = int(fileName.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i, :] = img2Vector('trainingDigits/' + fileName)

    return trainingMat, hwLabels


# 测试
def handWritingClassTest():
    trainingMat, hwLabels = getMatAndLabels()

    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    testLength = len(testFileList)
    for i in range(testLength):
        fileName = testFileList[i]
        classNum = int(fileName.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/' + fileName)
        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(fileName, '\t the classifier came back with: %d , the real answer is: %d' % (classifierResult, classNum))
        if classifierResult != classNum:
            errorCount += 1
    print('the total number of error is ', errorCount)
    print('the total error rate is ', errorCount / float(testLength))


if __name__ == '__main__':
    #handWritingClassTest()

    trainingMat, hwLabels = getMatAndLabels()
    fileName = '8_13.txt'
    classNum = int(fileName.split('_')[0])
    vectorUnderTest = img2Vector('testDigits/' + fileName)
    classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    print('the classifier came back with: %d , the real answer is: %d' % (classifierResult, classNum))
