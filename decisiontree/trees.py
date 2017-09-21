# (1)收集数据
# (2)准备数据
# (3)分析数据
# (4)训练算法
# (5)测试算法
# (6)使用算法
from math import log
import operator


# 计算信息熵
def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for data in dataSet:
        currLabel = data[-1]
        if currLabel not in labelCounts.keys():
            labelCounts[currLabel] = 0
        labelCounts[currLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries  # 概率
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 创建数据集
def createDataSet():
    dataSet = [
        [0, 0, 0, 'zero'],
        [0, 0, 1, 'one'],
        [0, 1, 0, 'two'],
        [0, 1, 1, 'three'],
        [1, 0, 0, 'four'],
        [1, 0, 1, 'five'],
        [1, 1, 0, 'six'],
        [1, 1, 1, 'seven']
        # [1, 1, 'yes'],
        # [1, 1, 'yes'],
        # [1, 0, 'no'],
        # [0, 1, 'no'],
        # [0, 1, 'no']
    ]
    labels = ['first', 'second', 'third']
    return dataSet, labels


# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 最优划分
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeature):
        featList = [example[i] for example in dataSet]  #
        uniqueVals = set(featList)  # 转set消除相同元素
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 找出次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [e[-1] for e in dataSet]
    if classList.count(classList[0]) == len(classList):  # 只剩一种类别
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征，返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    uniqueValues = set([e[bestFeat] for e in dataSet])
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    global classLabel
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 标签转index
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            classLabel = secondDict[key] if type(secondDict[key]) != dict \
                else classify(secondDict[key], featLabels, testVec)
    return classLabel


import pickle


def saveTree(tree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(tree, fw)


def grabTree(filename):
    import os
    if os.path.exists(filename):
        fr = open(filename,'rb')
        return pickle.load(fr)
    else:
        return None

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # shannonEnt=calShannonEnt(dataSet)
    # a = chooseBestFeatureToSplit(dataSet)
    # print(a)

    # for i in range(len(dataSet[0])):
    #     sData=splitDataSet(dataSet,i,'no')
    #     print(i,'\t',calShannonEnt(sData))
    tree = grabTree('tree.txt')
    if tree == None:
        tree = createTree(dataSet, labels.copy())
        saveTree(tree,'tree.txt')

    import decisiontree.treePlotter as pl
    # pl.createPlot(tree)

    testLabel = classify(tree, labels, [1, 0, 1])
    print(testLabel)
