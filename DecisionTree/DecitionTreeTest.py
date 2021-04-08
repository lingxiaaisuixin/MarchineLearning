#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2019/7/11 22:44
@Author :
@Desc :
"""
from math import log


def createDataSet():
    """
    :return: dataset,labels
    根据以下 2 个特征，将动物分成两类：鱼类和非鱼类。
    特征：
    不浮出水面是否可以生存
    是否有脚蹼
    """
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    :param dataSet:
    :return:
    计算给定数据集的香农熵的函数
    """
    # 求list的长度，表示计算参与训练的数据量
    numEnreies = len(dataSet)
    # 记录分类标签label出现的次数
    labelCounts = {}
    for featVec in dataSet:
        # 标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        # 为分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 对于 label 标签的占比，求出 label 标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key]) / numEnreies
        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


if __name__ == "__main__":
    myDat, labels = createDataSet()
    print(myDat)
    shannon = calcShannonEnt(myDat)
    print(shannon)
