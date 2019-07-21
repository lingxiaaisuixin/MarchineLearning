#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2019/7/21 18:19
@Author : SundialDreams
@Desc :绘制决策树
"""
#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
@datetime: 2019/7/12
@name:
@description:
"""
import operator
import numpy as np

# 导入数据
def createDataSet():
    dataSet = [['youth', 'no', 'no', 1, 'refuse'],
               ['youth', 'no', 'no', '2', 'refuse'],
               ['youth', 'yes', 'no', '2', 'agree'],
               ['youth', 'yes', 'yes', 1, 'agree'],
               ['youth', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', '2', 'refuse'],
               ['mid', 'yes', 'yes', '2', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '2', 'agree'],
               ['elder', 'yes', 'no', '2', 'agree'],
               ['elder', 'yes', 'no', '3', 'agree'],
               ['elder', 'no', 'no', 1, 'refuse'],            ]
    # print(type(dataSet))
    labels = ['age', 'working', 'house', 'credit_situation']
    return dataSet, labels

def calcuInfoEnt(dataSet, i=-1):
    '''
    计算信息熵
    dataSet:数据集
    return:数据集的信息熵
    '''

    numElements = len(dataSet)
    labelCounts = {}
    infoEnt = 0.0

    for elementVec in dataSet:  # 遍历数据集，统计元素向量中具有相同标签的频率
        currLabel = elementVec[i]
        if currLabel not in labelCounts.keys():
            labelCounts[currLabel] = 0
        labelCounts[currLabel] += 1

    for key in labelCounts:
        prob = float(labelCounts[key]) / numElements
        infoEnt -= prob * np.log2(prob)
    return infoEnt
def splitDataSet(dataSet, axis, featVal):
    '''
    按照给定特征值划分数据集
    dataSet:待划分数据集
    axis:划分数据集特征的维度
    featVal:特征的值
    return:划分的子数据集
    '''
    subDataSet = []
    for elementVec in dataSet:
        if elementVec[axis] == featVal:
            reduceElemVec = elementVec[:axis] #提取特征前的vec
            reduceElemVec.extend(elementVec[axis+1:]) #提取特征后的vec
            subDataSet.append(reduceElemVec)
    return subDataSet

def calcuConditionEnt(dataSet, i, featList, featSet):
    '''
    计算在指定特征i的条件下，Y的条件熵
    dataSet:数据集
    i:维度i
    featList:数据集特征值列表
    featSet:数据集特征值集合
    '''
    conditionEnt = 0.0
    for featVal in featSet:
        subDataSet = splitDataSet(dataSet, i, featVal)
        prob = float(len(subDataSet))/len(dataSet) #指定特征的概率
        conditionEnt += prob * calcuInfoEnt(subDataSet) #条件熵的定义计算
    return conditionEnt
def calcuInfoGain(dataSet, baseEnt, i):
    '''
    计算信息增益
    dataSet:数据集
    baseEnt:数据集的信息熵
    i:特征维度
    return:特征i对数据集的信息增益g(D|A)
    '''
    featList = [example[i] for example in dataSet] #第i维特征列表
    featSet  = set(featList) #转换为特征集合
    conditionEnt = calcuConditionEnt(dataSet, i, featList, featSet)
    infoGain = baseEnt - conditionEnt
    return infoGain
def chooseBestFeatSplitID3(dataSet):
    '''
    选择最好的数据集划分方式
    dataSet:数据集
    return:划分结果
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEnt = calcuInfoEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        infoGain = calcuInfoGain(dataSet, baseEnt, i)  # 计算信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最优特征维度


def majorityClassify(classList):
    '''
    采用多数表决的方法决定结点的分类
    classList:所有的类标签列表
    return:出现次数最多的类
    '''
    classCount = {}
    for cla in classList:
        if cla not in classCount.keys():
            classCount[cla] = 0
        classCount[cla] += 1
    sortClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                            reverse=True)
    return sortClassCount[0][0]


def crtDecisionTree(dataSet, featLabels):
    '''
    创建决策树
    dataSet:训练数据集
    featLabels:所有特征标签
    return：返回决策树字典
    '''
    classList = [element[-1] for element in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 所有的类标签都相同

    if len(dataSet[0]) == 1:
        return majorityClassify(classList)  # 用完所有特征

    bestFeat = chooseBestFeatSplitID3(dataSet)

    bestFeatLabel = featLabels[bestFeat]
    deTree = {bestFeatLabel: {}}

    subFeatLabels = featLabels[:]  # 复制所有类标签，保证每次递归调用时不改变原来的
    del (subFeatLabels[bestFeat])
    featValues = [element[bestFeat] for element in dataSet]
    featValSet = set(featValues)

    #####
    for value in featValSet:
        # subFeatLabels = featLabels[:]
        deTree[bestFeatLabel][value] = \
            crtDecisionTree(splitDataSet(dataSet, bestFeat, value), subFeatLabels)
    return deTree


import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="round", color='#3366FF')  # 定义判断结点形态
leafNode = dict(boxstyle="circle", color='#FF6633')  # 定义叶结点形态
arrow_args = dict(arrowstyle="<-", color='g')  # 定义箭头


# 计算叶子结点个数
def getNumLeafs(deTree):
    numLeafs = 0
    firstCondition = list(deTree.keys())[0]
    secondDict = deTree[firstCondition]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试结点的数据类型是否为字典
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 计算树的深度
def getTreeDepth(deTree):
    maxDepth = 0

    firstCondition = list(deTree.keys())[0]
    secondDict = deTree[firstCondition]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


# 绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xytext=centerPt, xycoords='axes fraction',
                            textcoords='axes fraction', va="center",
                            ha="center", bbox=nodeType, arrowprops=arrow_args)


# 在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center",
                        ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  # 计算宽与高
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))
              / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 标记子结点属性值
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 减少y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key],
                     (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree,
# and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def ContactLense():
    fr = open("data/lenses.txt")
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lensesTree = crtDecisionTree(lenses,lensesLabels)
    print(lensesTree)
    return lensesTree
# 测试代码
if __name__ == "__main__":
    lesenseTree = ContactLense()
    createPlot(lesenseTree)
    dataSet, featLabels = createDataSet()
    deTree = crtDecisionTree(dataSet, featLabels)
    createPlot(deTree)