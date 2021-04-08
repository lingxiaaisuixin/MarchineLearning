#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2020/12/3 21:25
@Author : StarsDreams
@Desc :
"""
import numpy as np

def loadDataSet():
    """
    主要功能时读取testSet.txt数据，每行前两个值为X1和X2，
    第三个值对应的是类别标签，为了方便计算该函数还将X0的值设为1.0
    :return:训练集和标签
    """
    dataMat = []
    labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    """
    sigmoid函数实现
    :param inX:
    :return:
    """
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
    梯度上升法
    :param dataMatIn: [X0,X1,X2]
    :param classLabels: [lable]
    :return: 回归系数
    """
    #转换成矩阵
    dataMatrix = np.mat(dataMatIn)             #convert to NumPy matrix
    #转换成矩阵，并转置
    labelMat = np.mat(classLabels).transpose() #convert to NumPy matrix
    #行列数
    m,n = np.shape(dataMatrix)
    #学习率
    alpha = 0.001
    #迭代次数
    maxCycles = 500
    # 构建回归系数
    weights = np.ones((n,1))
    #迭代求取回归系数
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        #误差
        error = (labelMat - h)              #vector subtraction
        #求解公式w:=w+alpha*x^T*(y(w*x)-y)
        #x^T表示x的转置，y(w*x)-y为预测误差
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def plotBestFit(weights):
    """
    画出决策边界
    :param weights:回归系数
    :return:
    """
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    """
    设置sigmoid函数为0,0为两个类别0和1的分界处
    则 0=w0*x0+w1*x1+w2*x2，其中x0=1
    这里未知数有两个，设x1= np.arange(-3.0, 3.0, 0.1),
    则x2 = (-w0-w1*x)/w2就画出了一条直线
    """

    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升
    一次仅用一个样本点更新回归系数
    :param dataMatrix: 样本集
    :param classLabels: 标签
    :return:
    """
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进随机梯度上升
    :param dataMatrix:样本集
    :param classLabels:标签
    :param numIter:迭代次数
    :return:回归系数
    """
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(np.sum(dataMatrix[dataIndex[randIndex]]*weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del(dataIndex[randIndex])
    return weights

def testing_effect():
    """
    回归梯度提升测试
    :return:
    """
    dataArr,labelMat = loadDataSet()
    weights = gradAscent(dataArr,labelMat )
    print(weights)

def testing_effect0():
    """
    画出决策边界测试
    :return:
    """
    dataArr,labelMat = loadDataSet()
    weights = gradAscent(dataArr,labelMat )
    plotBestFit(np.array(weights))

def testing_effect1():
    """
    随机梯度上升测试
    :return:
    """
    dataArr,labelMat = loadDataSet()
    weights = stocGradAscent0(np.array(dataArr),labelMat )
    plotBestFit(np.array(weights))

def testing_effect2():
    """
    改进的随机梯度上升测试
    :return:
    """
    dataArr,labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataArr),labelMat ,500)
    plotBestFit(np.array(weights))

# +++++++从疝气病症预测病马的死亡率+++++++
def classifyVector(inX, weights):
    """
    根据回归系数和特征向量来计算对应的Sigmoid的值，大于0.5函数返回1，否则返回0
    :param inX:特征向量
    :param weights:回归系数和
    :return:分类结果
    """
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    """
    用于打开训练集和测试集，并对数据进行格式化处理的函数
    :return:
    """
    frTrain = open('data\horseColicTraining.txt')
    frTest = open('data\horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    """
    调用函数colicTest() 10次并求结果的平均值
    :return:
    """
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def plot_tanh():
    """
    画tanh函数
    :return:
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    x = np.linspace(-10, 10)
    y = tanh(x)

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.set_yticks([-1, -0.5, 0.5, 1])

    plt.plot(x, y, label="Sigmoid", color="red")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # testing_effect()
    # testing_effect0()
    testing_effect1()
    # testing_effect2()
    # multiTest()
    # plot_tanh()
