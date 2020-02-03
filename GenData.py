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


for i in range(len(StockList)):
    stock1 = StockList[i]
    print(stock1)
    path1 = tweetpath  + stock1
    files1= os.listdir(path1)
    #tw1 = pd.DataFrame()
    tw1 = Gentw(path1, files1)
    for j in range(i + 1, len(StockList)):
        stock2 = StockList[j]
        #tw2 = pd.DataFrame()
        if stock2 != stock1:
            print(stock2)
            path2 = tweetpath  + stock2
            files2= os.listdir(path2)
            tw2 = Gentw(path2, files2)
            df = GenProfit(tw1, tw2, stock1, stock2, i, j)
            len_train = int(len(df) * 0.8)
            len_val = int(len(df) * 0.65)
            train1 = df[:len_val]
            validation1 = df[len_val:len_train]
            test1 = df[len_train:]
            train = pd.concat([train, train1]).reset_index(drop=True)
            validation = pd.concat([validation, validation1]).reset_index(drop=True)
            test = pd.concat([test, test1]).reset_index(drop=True)
            print(train)
            
            
train = deal(train)
test = deal(test)
validation = deal(validation)

def deal_dir1(df):
    df['num'] = 2
    df = df[['front_testid', 'front_features', 'labels_index1', 'num']]
    df.columns = ['testid', 'features_content', 'labels_index', 'labels_num']
    df['testid'] = df['testid'].astype(int)
    return df

def deal_dir2(df):
    df['num'] = 2
    df = df[['behind_testid', 'behind_features', 'labels_index2', 'num']]
    df.columns = ['testid', 'features_content', 'labels_index', 'labels_num']
    df['testid'] = df['testid'].astype(int)
    return df

train1 = deal_dir1(train)
validation1 = deal_dir1(validation)
test1 = deal_dir1(test)
test2 = deal_dir2(test)

train1.to_json('data/Train.json', orient='records', lines=True)
validation1.to_json('data/Validation.json', orient='records', lines=True)

test1.to_json('data/Test1.json', orient='records', lines=True)
test2.to_json('data/Test2.json', orient='records', lines=True)


        
