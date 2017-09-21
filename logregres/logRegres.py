from math import exp
from numpy import *


def loadData():
    dataMat, labelMat = [], []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()  # mat.T
    m, n = shape(dataMatrix)  # 100，3
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 100，1
        error = labelMat - h  # 100，1
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


import matplotlib.pyplot as plt


def showFigure(dataMat, classMat):
    classMat = classMat + ones(len(classMat))  # 防止0不显示

    dataMat = array(dataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 1], dataMat[:, 2], 15.0 * array(classMat), 15.0 * array(classMat))
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.show()


def plotBestFit(weights):
    dataMat, labelMat = loadData()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xCord1, yCord1 = [], []
    xCord2, yCord2 = [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xCord1.append(dataArr[i, 1])
            yCord1.append(dataArr[i, 2])
        else:
            xCord2.append(dataArr[i, 1])
            yCord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xCord1, yCord1, s=30, c='red', marker='s')
    ax.scatter(xCord2, yCord2, s=30, c='green')
    x = arange(start=-3.0, stop=3.0, step=0.1)
    y = array((-weights[0] - weights[1] * x) / weights[2])[0]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadData()
    # showFigure(dataMat, labelMat)

    plotBestFit(gradAscent(dataMat, labelMat))
