#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2020/11/29 11:04
@Author : StarsDreams
@Desc :
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 读取数据集
dataset = fetch_20newsgroups(subset='all')
data, target = dataset.data, dataset.target
# 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)
# 特征抽取
tf_idf = TfidfVectorizer()
x_train = tf_idf.fit_transform(x_train)

x_test = tf_idf.transform(x_test)

# 使用朴素贝叶斯进行分类
nbclf = MultinomialNB(alpha=1.0)
nbclf.fit(x_train, y_train)
y_predict = nbclf.predict(x_test)

print("预测的文本类别为：", y_predict)
print("分类分数为：", nbclf.score(x_test, y_test))

