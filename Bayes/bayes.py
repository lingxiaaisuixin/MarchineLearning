#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2020/11/26 20:54
@Author : StarsDreams
@Desc :
"""

from numpy import *
import re
import random


def loadDataSet():
    """
    创建了一些实验样本
    :return:
    postingList是进行词条切割后的文档集合（文档来自斑点犬爱好者留言板）
    classVec是一个类别标签集合，有两个类别侮辱性和非侮辱性。
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 代表侮辱性的文字, 0 代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建一个包含所有文档中出现的不重复词列表，使用set集合
    :param dataSet:一个数据集合
    :return:返回一个不重复词表
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 合并两个set集合
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    将输入的文档转化成向量
    :param vocabList:词汇表
    :param inputSet:某个文档
    :return:文档向量
    """
    # 创建一个和词汇表等长的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def testing_efforts():
    """
    测试效果
    :return: None
    """
    # 加载数据集
    listOPosts, list_Classes = loadDataSet()
    # 创建词汇表
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    print(setOfWords2Vec(myVocabList, listOPosts[0]))
    print(setOfWords2Vec(myVocabList, listOPosts[3]))


def trainNB0ld(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器
    :param trainMatrix:文档矩阵
    :param trainCategory:文档类别标签
    :return:返回属于侮辱性文档的概率和两个类别的概率向量
    """
    # 文档数
    numTrainDocs = len(trainMatrix)
    # 单词数
    numWords = len(trainMatrix[0])
    # 计算文档属于侮辱性文档（class=1）的概率，p(1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建两个值全为0数组，记录单词出现次数
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)  # change to ones()
    p0Denom = 0.0
    p1Denom = 0.0  # change to 2.0
    for i in range(numTrainDocs):
        # 是否是侮辱性文档
        if trainCategory[i] == 1:
            # 如果是侮辱性文档，对侮辱性文档的向量进行相加
            p1Num += trainMatrix[i]
            # 目的计算所有侮辱性文档中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            # 如果是非侮辱性文档，对侮辱性文档的向量进行相加
            p0Num += trainMatrix[i]
            # 目的计算所有非侮辱性文档中出现的单词总数
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom  # change to log()
    p0Vect = p0Num / p0Denom  # change to log()
    return p0Vect, p1Vect, pAbusive


def testing_efforts0():
    """
    测试贝叶斯分类
    :return: None
    """
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0ld(array(trainMat), array(listClasses))
    print(p0V, '\n', p1V, '\n', pAb)


def trainNB0(trainMatrix, trainCategory):
    """
    改进后的朴素贝叶斯，因为在分类时要计算多个概率的乘积来获得某个文档属于某个类别的概率
    即计算p(w0|1)p(w1|1)p(w2|1)...如果其中有一个概率为0，最后乘积为0，显然是不合理的，为
    降低这种影响，将词的出现次数初始化为1 ，如将p0Num和p1Num初始化为1的数组，将每个类别
    出现的总词数设置为2，如p0Denom和p1Denom设置为2
    :param trainMatrix:文档矩阵
    :param trainCategory:文档对应的类别
    :return:
    """
    # 总文档数
    numTrainDocs = len(trainMatrix)
    # 总单词数
    numWords = len(trainMatrix[0])
    # 侮辱性文档出现的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建两个值全为1数组，记录单词出现次数
    p0Num = ones(numWords);
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0;
    p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 如果是侮辱性文档，对侮辱性文档的向量进行相加
            p1Num += trainMatrix[i]
            # 目的计算所有侮辱性文档中出现的单词总数
            p1Denom += sum(trainMatrix[i])
        else:
            # 如果是非侮辱性文档，对侮辱性文档的向量进行相加
            p0Num += trainMatrix[i]
            # 目的计算所有非侮辱性文档中出现的单词总数
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    得出分类结果
    :param vec2Classify:要分类的向量
    :param p0Vec:类别为0的概率向量
    :param p1Vec:类别为1的概率向量
    :param pClass1:侮辱性文档的概率
    :return:返回最大该利率对应的类别标签
    """
    # 预测文档向量
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    测试数据集
    :return: None
    """
    # 加载数据集
    listOPosts, listClasses = loadDataSet()
    # 单词集合
    myVocabList = createVocabList(listOPosts)
    # 数据向量化
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 两个类别的概率向量和侮辱性文档的概率
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # 测试文档
    testEntry = ['love', 'my', 'dalmation']
    # 转换成词向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # 判断文档类别
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def test_split():
    mySent = 'This book is the best book on python or M.L. I have ever laid eyes upon.'
    print(mySent.split())

    # regEx = re.compile('\\W* ')
    regEx = re.compile("\\W")
    listofTokens = regEx.split(mySent)
    print(listofTokens)
    print([tok.lower() for tok in listofTokens if len(tok) > 0])
    emailText = open('data\\email\\ham\\6.txt').read()
    listofTokens = regEx.split(emailText)
    print(listofTokens)
    print([tok.lower() for tok in listofTokens if len(tok) > 0])


###垃圾邮件测试
def textParse(bigString):  # input is big string, #output is word list
    """
    实现分词功能,并去掉少于两个字符的字符串，并将字符转为小写
    :param bigString:邮件文本内容
    :return:全部是小写的word列表，去掉少于 2 个字符的字符串
    """
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('data\\email\\spam\\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('data\\email\\ham\\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    # create test set
    trainingSet = list(range(50))
    # 取10个随机整数，范围[0，50）
    testSet = random.sample(range(50), 10)
    # 获取训练集
    trainingSet = list(set(trainingSet) - set(testSet))
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


# ----- 示例: 使用朴素贝叶斯分类器从个人广告中获取区域倾向 ------
def calcMostFreq(vocabList, fullText):
    """
    统计词频，并返回最高的30个单词
    :param vocabList: 词汇表
    :param fullText: 文本
    :return: 词频最高的30个单词
    """
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    """
    加载数据
    :param feed1:
    :param feed0:
    :return:
    """
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # 构造特征和标签
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词汇表
    vocabList = createVocabList(docList)  # create vocabulary
    # 去掉高频词
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    # create test set
    trainingSet = list(range(2 * minLen))
    # 取20个随机整数
    testSet = random.sample(range(2 * minLen), 20)
    # 获取训练集
    trainingSet = list(set(trainingSet) - set(testSet))
    trainMat = []
    trainClasses = []
    # 转成向量
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 分类
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def testing_rss():
    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocab_list, p_sf, p_nf = localWords(ny, sf)
    print(vocab_list, p_sf, p_nf)


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


def testing_spamTest():
    spamTest()


if __name__ == "__main__":
    testing_efforts()
    # testing_efforts0()
    # testingNB()
    # test_split()
    # testing_spamTest()
    # testing_rss()
