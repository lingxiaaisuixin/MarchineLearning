#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2021/3/1 21:11
@Author : StarsDreams
@Desc :
"""
from numpy import *


def loadDataSet(filename):
    """
    读取数据集
    :param filename:
    :return:
    """
    dataMat = [];
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    :param i: alpha的下标
    :param m: 所有alpha的数目
    :return:返回一个不为i的随机数，在[0,m)之间的整数
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """
    用于调整于H或小于h的alpha值
    :param aj:
    :param H:
    :param L:
    :return:
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前的最大循环次数
    :return:
    """
    # 将数据集和标签转成NumPy矩阵，方便数学计算
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    # 构建alpha列矩阵
    alphas = mat(zeros((m, 1)))
    #用于记录在没有任何alpha改变的情况下遍历数据集的次数
    iter = 0
    while (iter < maxIter):
        # 用于记录alpha是否已经进行优化
        alphaPairsChanged = 0
        for i in range(m):
            # fXi预测的类别 Ei计算误差
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])  # if checks if an example violates KKT conditions
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 不管正间隔还是负间隔都会被测试，alpha值不能为0和C,因为此时点已经在边界上了，不能再进行优化了
                # 随机选择alpha的值alphas[j]
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 记录旧的alpha值，用于跟新的alhpa值比较
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算L、H,用于将alphas[i]调整到0到C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                #eta是alphas[j]的最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j,:] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查alphas[j]是否有变化
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # alphas[j]如果有变化，alphas[i]、alphas[j]同时变化，改变的大小一致但是方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # update i by the same amount as j
                # the update is in the oppostie direction
                # 对 alphas[i]、alphas[j]优化后，设置查常数
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet('data/testSet.txt')
    print(dataMat, labelMat)
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.01, 40)
