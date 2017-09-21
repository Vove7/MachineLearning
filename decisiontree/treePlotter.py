import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                           xytext=centerPt, textcoords='axes fraction',
                           va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            thisDepth = 1 + getNumLeafs(secondDict[key])
        else:
            thisDepth = 1
        maxDepth = thisDepth if maxDepth < thisDepth else maxDepth
    return maxDepth


def plotMidText(cnPt, parentPt, nodeText):
    xMid = (parentPt[0] - cnPt[0] / 2.0) + cnPt[0]
    yMid = (parentPt[1] - cnPt[1] / 2.0) + cnPt[1]
    createPlot.ax.text(xMid, yMid, nodeText)


def plotTree(myTree, parentPt, nodeText):
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cnPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cnPt, parentPt, nodeText)
    plotNode(firstStr, cnPt, parentPt, decisionNode)

    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            plotTree(secondDict[key], cnPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cnPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cnPt, str(key))

    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



def createPlot(inTree):
    flg = plt.figure(1, facecolor='white')
    flg.clf()
    # axprops=dict(xticks=[],yticks=[])
    createPlot.ax = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), ' ')
    plt.show()
