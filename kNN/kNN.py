from numpy import *
import operator

#创建数据集
def createDataset():
    group = array(([1., 1.1], [1., 1.], [0, 0], [0, 0.1]))
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSize, 1)) - dataSet
    diss = (diffMat ** 2).sum(axis=1) ** 0.5
    sortDiss = diss.argsort()

    classcount = {}
    for i in range(k):
        votelabel = labels[sortDiss[i]]
        classcount[votelabel] = classcount.get(votelabel, 0) + 1
    sortclasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sortclasscount[0][0]


if __name__ == '__main__':
    g, l = createDataset()
    a = classify0([0, 0], g, l, 3)
    print(a)