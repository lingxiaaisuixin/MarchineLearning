#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time  : 2020/2/18 11:48
@Author : StarsDreams
@Desc :
"""

# coding: utf-8

# In[ ]:

import pandas as pd
import os

##########################################################################################################################################################################################
# # 1 数据整理

# ## 1.1导入数据

# In[2]:
path = r'C:\Users\38681\Desktop\bankcredit'
filelist = os.listdir(path)
# 导入数据
dataVar = locals()
for file in filelist:
    if file.endswith('csv'):
        filepath = path +'\\'+ file
#         print(filepath)
        dataVar[file.split('.')[0]] = pd.read_csv(filepath,encoding = 'gbk')
#         print(file.split('.')[0])
print(dataVar)
# ## 1.2、生成被解释变量bad_good

# In[3]:

bad_good = {'B': 1, 'D': 1, 'A': 0, 'C': 2}
loans['bad_good'] = loans.status.map(bad_good)
loans.head()

# ## 1.3、借款人的年龄、性别

# In[4]:

data2 = pd.merge(loans, disp, on='account_id', how='left')
data2 = pd.merge(data2, clients, on='client_id', how='left')
data2 = data2[data2.type == '所有者']

data2.head()
# ## 1.4、借款人居住地的经济状况

# In[5]:

data3 = pd.merge(data2, district, left_on='district_id', right_on='A1', how='left')
data3.head()

# ## 1.5、贷款前一年内的账户平均余额、余额的标准差、变异系数、平均收入和平均支出的比例

# In[6]:

data_4temp1 = pd.merge(loans[['account_id', 'date']],
                       trans[['account_id', 'type', 'amount', 'balance', 'date']],
                       on='account_id')
data_4temp1.columns = ['account_id', 'date', 'type', 'amount', 'balance', 't_date']
data_4temp1 = data_4temp1.sort_values(by=['account_id', 't_date'])

data_4temp1['date'] = pd.to_datetime(data_4temp1['date'])
data_4temp1['t_date'] = pd.to_datetime(data_4temp1['t_date'])
data_4temp1.tail()

# ## 将对账户余额进行清洗
# In[9]:

data_4temp1['balance2'] = data_4temp1['balance'].map(
    lambda x: int(''.join(x[1:].split(','))))
data_4temp1['amount2'] = data_4temp1['amount'].map(
    lambda x: int(''.join(x[1:].split(','))))

data_4temp1.head()

# ## 根据取数窗口提取交易数据
# In[10]:

import datetime

data_4temp2 = data_4temp1[data_4temp1.date > data_4temp1.t_date][
    data_4temp1.date < data_4temp1.t_date + datetime.timedelta(days=365)]
data_4temp2.tail()

# ### 1.5.1账户平均余额、余额的标准差、变异系数

# In[11]:

data_4temp3 = data_4temp2.groupby('account_id')['balance2'].agg([('avg_balance', 'mean'), ('stdev_balance', 'std')])
data_4temp3['cv_balance'] = data_4temp3[['avg_balance', 'stdev_balance']].apply(lambda x: x[1] / x[0], axis=1)

data_4temp3.head()

# ### 1.5.2 平均支出和平均收入的比例
# In[12]:

type_dict = {'借': 'out', '贷': 'income'}
data_4temp2['type1'] = data_4temp2.type.map(type_dict)
data_4temp4 = data_4temp2.groupby(['account_id', 'type1'])[['amount2']].sum()
data_4temp4.head()

# In[13]:

data_4temp5 = pd.pivot_table(
    data_4temp4, values='amount2',
    index='account_id', columns='type1')
data_4temp5.fillna(0, inplace=True)
data_4temp5['r_out_in'] = data_4temp5[
    ['out', 'income']].apply(lambda x: x[0] / x[1], axis=1)
data_4temp5.head()

# In[14]:

data4 = pd.merge(data3, data_4temp3, left_on='account_id', right_index=True, how='left')
data4 = pd.merge(data4, data_4temp5, left_on='account_id', right_index=True, how='left')

data4.head()
# ## 1.6、计算贷存比，贷收比

# In[15]:

data4['r_lb'] = data4[['amount', 'avg_balance']].apply(lambda x: x[0] / x[1], axis=1)
data4['r_lincome'] = data4[['amount', 'income']].apply(lambda x: x[0] / x[1], axis=1)

data4.head()
# In[17]:
##########################################################################################################################################################################################
# # 2 构建Logistic模型
data4.columns
# •提取状态为C的用于预测。其它样本随机抽样，建立训练集与测试集

# In[17]:

data_model = data4[data4.status != 'C']
for_predict = data4[data4.status == 'C']

train = data_model.sample(frac=0.7, random_state=1235).copy()
test = data_model[~ data_model.index.isin(train.index)].copy()
print(' 训练集样本量: %i \n 测试集样本量: %i' % (len(train), len(test)))


# In[22]:

# 向前法
def forward_select(data, response):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response, ' + '.join(selected + [candidate]))
            aic = smf.glm(
                formula=formula, data=data,
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = aic_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print('aic is {},continuing!'.format(current_score))
        else:
            print('forward selection over!')
            break

    formula = "{} ~ {} ".format(response, ' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.glm(
        formula=formula, data=data,
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return (model)


# In[23]:

candidates = ['bad_good', 'A1', 'GDP', 'A4', 'A10', 'A11', 'A12', 'amount', 'duration',
              'A13', 'A14', 'A15', 'a16', 'avg_balance', 'stdev_balance',
              'cv_balance', 'income', 'out', 'r_out_in', 'r_lb', 'r_lincome']
data_for_select = train[candidates]

lg_m1 = forward_select(data=data_for_select, response='bad_good')
lg_m1.summary().tables[1]


#定义向前逐步回归函数
# def forward_select(data,target):
#     variate=set(data.columns)  #将字段名转换成字典类型
#     variate.remove(target)  #去掉因变量的字段名
#     selected=[]
#     current_score,best_new_score=float('inf'),float('inf')  #目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
#     #循环筛选变量
#     while variate:
#         aic_with_variate=[]
#         for candidate in variate:  #逐个遍历自变量
#             formula="{}~{}".format(target,"+".join(selected+[candidate]))  #将自变量名连接起来
#             aic=ols(formula=formula,data=data).fit().aic  #利用ols训练模型得出aic值
#             aic_with_variate.append((aic,candidate))  #将第每一次的aic值放进空列表
#         aic_with_variate.sort(reverse=True)  #降序排序aic值
#         best_new_score,best_candidate=aic_with_variate.pop()  #最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
#         if current_score>best_new_score:  #如果目前的aic值大于最好的aic值
#             variate.remove(best_candidate)  #移除加进来的变量名，即第二次循环时，不考虑此自变量了
#             selected.append(best_candidate)  #将此自变量作为加进模型中的自变量
#             current_score=best_new_score  #最新的分数等于最好的分数
#             print("aic is {},continuing!".format(current_score))  #输出最小的aic值
#         else:
#             print("for selection over!")
#             break
#     formula="{}~{}".format(target,"+".join(selected))  #最终的模型式子
#     print("final formula is {}".format(formula))
#     model=ols(formula=formula,data=data).fit()
#     return(model)

# In[24]:

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

fpr, tpr, th = metrics.roc_curve(test.bad_good, lg_m1.predict(test))
plt.figure(figsize=[6, 6])
plt.plot(fpr, tpr, 'b--')
plt.title('ROC curve')
plt.show()

# In[25]:

print('AUC = %.4f' % metrics.auc(fpr, tpr))

# In[28]:

for_predict['prob'] = lg_m1.predict(for_predict)
for_predict[['account_id', 'prob']].head()

# In[ ]:
