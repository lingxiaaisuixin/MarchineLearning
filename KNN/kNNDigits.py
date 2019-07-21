#!/usr/bin/python
#-*-coding:utf-8-*-
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    """l
    :param inX:用于分类的输入向量/测试数据
    :param dataSet:训练数据集的 features
    :param labels:训练数据集的 labels
    :param k:选择最近邻的数目
    :return:sortedClassCount[0][0] -- 输入向量的预测分类 labels
    """
    """
    分类主体程序，计算欧式距离，选择距离最小的k个，返回k个中出现频率最高的类别
    inX是所要测试的向量
    dataSet是训练样本集，一行对应一个样本。dataSet对应的标签向量为labels
    k是所选的最近邻数目
    """
    # 数据集的大小
    dataSetSize = dataSet.shape[0]
    #函数tile(A,reps) A为原数组
    """
    假设reps为（a,b,c,d,e,f） 数字从右到左，数组维度从最深维度到最低维度,则数组最深维度重复f次；
    然后次深维度重复e次；接着次次深维度重复d次；再然后次次次深维度重复c次…… 
    以此类推，直到对最低维度重复a次。
     >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])
    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])
    """
    # 距离度量，度量公式欧氏距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # 取平方
    sqDiffMat = diffMat**2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances**0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,5]) 则y.argsort() = [1 3 2 0 4]
    sortedDistIndicies = distances.argsort()            
    classCount={}
    # 选择距离最小的k的个数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        """
        如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        data = {5:2,3:4}
        data.get(3,0)返回的值是4；
        data.get（1,0）返回值是0；
        """
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    """
    字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
    """
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
def img2vector(filename):
    """
    将文件中的图片信息转化为1*1024的向量。
    :param filename: 文件名称，如0_1.txt
    :return: returnVect 返回一个1*1024的向量
    """
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    """
    将训练集图片合并成100 * 1024的大矩阵
    同时逐一对测试集中的样本分类
    """
    hwLabels = []
    trainingFileList = listdir('data/trainingDigits')
    m = len(trainingFileList)
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]                  
        fileStr = fileNameStr.split('.')[0]                
        classNumStr = int(fileStr.split('_')[0])          
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('data/trainingDigits/%s' % fileNameStr)
     
    testFileList = listdir('data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
    print("\nthe total accuracy rate is: %f" % ((mTest-errorCount)/mTest*100),'%')
if __name__=="__main__":
    handwritingClassTest()
