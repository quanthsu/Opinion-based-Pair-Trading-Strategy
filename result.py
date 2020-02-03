import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import json
import os

import time
import datetime
from sklearn import preprocessing

import keras
MAX_NUM_WORDS = 10000
tokenizer = keras \
    .preprocessing \
    .text \
    .Tokenizer(num_words=MAX_NUM_WORDS)

import time
import datetime
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from tojson import *


tweetpath = '../stocknet-code/stocknet-dataset/tweet/raw/'
pricepath = '../stocknet-code/stocknet-dataset/price/raw/'


tw = pd.DataFrame()
path = tweetpath
StockList= os.listdir(path)
StockList.sort()

train = pd.DataFrame()
test = pd.DataFrame()
validation = pd.DataFrame()



StockList = ['AMZN', 'BABA', 'WMT', 'CMCSA', 'HD', 'DIS', 'MCD', 'CHTR', 'UPS', 'PCLN'] #Service


def ComputePro1(com):
    profit = 0
    plotpro = {}
    for i in range(len(com)):
        s = str(com['front_testid'][i])
        s_datetime = datetime.datetime.strptime(s, '%Y%m%d')
        if int(com['predict_labels'][i]) == 1 :
            profit += com['label'][i]
            if s_datetime in plotpro.keys():
                plotpro[s_datetime] += com['label'][i]
            else:
                plotpro[s_datetime] = com['label'][i]
            
        else:
            profit -= com['label'][i]
            if s_datetime in plotpro.keys():
                plotpro[s_datetime] -=  com['label'][i]
            else:
                plotpro[s_datetime] = -com['label'][i]
        
         
    print('Profit : ', profit)
    
    return profit, plotpro
    

def ComProDir1(test, pro):
    profit = 0
    plotdir = {}
    for i in range(len(pro)):
        s = str(pro['id'][i])
        s_datetime = datetime.datetime.strptime(s, '%Y%m%d')
        if (pro['predict_labels'][i] == [1]) & (pro['predict_labels1'][i] == [0]):
            profit += test['label'][i]
            if s_datetime in plotdir.keys():
                plotdir[s_datetime] += test['label'][i]
            else:
                plotdir[s_datetime] = test['label'][i]
                
        elif (pro['predict_labels'][i] == [0]) & (pro['predict_labels1'][i] == [1]):
            profit -= test['label'][i]
            if s_datetime in plotdir.keys():
                plotdir[s_datetime] -= test['label'][i]
            else:
                plotdir[s_datetime] = -test['label'][i]
            
    print (profit)
    
    return profit, plotdir

def pl(plotpp):
    pp = pd.DataFrame(plotpp,index=[0]).reset_index(drop=True)
    pp = pp.T
    pp.columns = ['profit']
    pp['Date'] = pp.index
    pp = pp.sort_values('Date')
    #pp.index = pp['Date']
    
    return pp 

with open('1CNN/results/1578582984/predictions.json') as f:
    data = json.loads("[" + 
        f.read().replace("}\n{", "},\n{") + 
    "]")
    
d = pd.DataFrame(data)
acc = d[d['labels'] == d['predict_labels']]

print('Accuracy : ', len(acc) / len(d))


with open('1CNN/results/1578582984/predictions.json') as f:
    data = json.loads("[" + 
        f.read().replace("}\n{", "},\n{") + 
    "]")
    
d1 = pd.DataFrame(data)
acc = d1[d1['labels'] == d1['predict_labels']]
d1.rename(columns={'predict_labels':'predict_labels1'}, inplace = True)

print('Accuracy : ', len(acc) / len(d1))


pro = pd.concat([d, d1], axis=1)
pro = pro.loc[:,~pro.columns.duplicated()]
p2, plotdir = ComProDir1(test, pro)

with open('CNN/results/1577808713/predictions.json') as f:
    data = json.loads("[" + 
        f.read().replace("}\n{", "},\n{") + 
    "]")
    
d = pd.DataFrame(data)

print ('Accuracy : ', accuracy_score(d['labels'], d['predict_labels']))
sum += accuracy_score(d['labels'], d['predict_labels'])
print('precision_score : ', precision_score(d['labels'], d['predict_labels']))
print ('F1 score : ', f1_score(d['labels'], d['predict_labels']))
print('recall_score : ', recall_score(d['labels'], d['predict_labels']))
print('confusion_matrix : \n', confusion_matrix(d['labels'], d['predict_labels']))

#com = pd.concat([d, test], axis = 1)
#p += ComputePro(com)
com = pd.concat([d, test], axis = 1)
com = com.loc[:,~com.columns.duplicated()]
p1, plotpro = ComputePro1(com)
p += p1


pp = pl(plotdir)
pp2 = pl(plotpro)

pp = pd.merge(pp, pp2, left_index=True, right_index=True, how='inner')

SP = pd.read_csv('data/^GSPC.csv', index_col=0)

SP = SP.pct_change() *100000

res = pd.merge(pp, SP, left_index=True, right_index=True, how='inner')


res['profit_x'] /= 1000
res['profit_y'] /= 1000
res['Adj Close'] /= 1000



plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False  

plt.figure(figsize=(10, 7))
plt.grid(linestyle="--")  
ax = plt.gca()
ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False)  

ax.plot(ans['profit_cnn_ind'], label='CNN-IND')
ax.plot(ans['profit_cnn_pw'], label='CNN-PW')
ax.plot(ans['profit_rnn_ind'], label='RNN-IND')
ax.plot(ans['profit_rnn_pw'], label='RNN-PW')
ax.plot(ans['profit_han_ind'], label='HAN-IND')
ax.plot(ans['profit_han_pw'], label='HAN-PW')
ax.plot(ans['profit_sann_ind'], label='SANN-IND')
ax.plot(ans['profit_sann_pw'], label='SANN-PW')
ax.plot(ans['profit_abcnn_pw'], label='ABCNN-PW', color = 'red', linestyle="-." )
ax.plot(ans['profit_rand'], label='Random')
ax.plot(ans['SP'], label='S&P500', linestyle="--")


plt.xlabel("Date", fontsize=13, fontweight='bold')
plt.ylabel("Return On Investment (%)", fontsize=13, fontweight='bold')
   
# plt.legend()         
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')  

plt.show()
        
