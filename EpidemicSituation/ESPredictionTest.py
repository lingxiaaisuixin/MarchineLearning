#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2020/2/12 10:04
@Author : StarsDreams
@Desc :
"""
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel(r'C:\Users\38681\Desktop\epidemic Situation\疫情数据城市.xlsx')
data = data[11:]
tempdate = data['date'].apply(lambda x : x.strftime('%m-%d'))
print(tempdate,type(data['date']))

tday = datetime.date.today()
print(tday)
print(type(tday))
basedate = '2020-01-21'
basedate00 = datetime.date(*map(int, basedate.split('-')))
print('basedate00:',basedate00,type(basedate00))
basedate = time.strptime(basedate,"%Y-%m-%d")
print('basedate:',type(basedate))
tdelta = datetime.timedelta(days=1)
print(tday+tdelta)
days = np.linspace(0,9,10)
y = [(2*d-3) for d in days]
print(y)
print(days)
days_all=[]
for d in days:
    print(d)
    tempdelta = datetime.timedelta(days=d)
    temp = tday + tempdelta
    print(temp.strftime('%m-%d'))
    days_all.append(tday+tempdelta)
print(days_all)
plt.plot(days_all,y)
plt.xticks(rotation=90)
plt.show()
