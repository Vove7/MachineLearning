import matplotlib.pyplot as plt
from numpy import *

import kNN


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    reMatrix = zeros((len(arrayOLines), 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        reMatrix[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return reMatrix, classLabelVector


# 归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normalDataSet = zeros(shape(dataSet))
    normalDataSet = dataSet - tile(minVals, (m, 1))
    normalDataSet /= tile(ranges, (m, 1))
    return normalDataSet, ranges, minVals


# 测试
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(normMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = kNN.classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                         datingLabels[numTestVecs:m], 3)
        print("the classify call back with : %d ,the real answer is % d "
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print("the total error rate is %f " % (errorCount / float(numTestVecs)))


# 正式预测
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent filter miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])

    classifierResult = kNN.classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)

    print('You will probably like this person :', resultList[classifierResult - 1])


# 绘制散点图
def showFigure():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    datingDataMat = autoNorm(datingDataMat)[0]

    # 显示散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(223)

    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15 * array(datingLabels), 15 * array(datingLabels))
    plt.show()


if __name__ == '__main__':
    # classifyPerson()
    showFigure()
    # datingClassTest()
