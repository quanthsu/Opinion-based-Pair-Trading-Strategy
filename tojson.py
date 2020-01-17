import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

import time
import datetime
from sklearn import preprocessing

import keras

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

tweetpath = '../stocknet-code/stocknet-dataset/tweet/raw/'
pricepath = '../stocknet-code/stocknet-dataset/price/raw/'

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

text_remove = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
#text = re.sub(text_remove, ' ', str(text).lower())
def clean_text(text):
    text = re.sub(text_remove,' ',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    #text = nltk.word_tokenize(str(text))
    text = tknzr.tokenize(str(text))
    text = " ".join(text)
    return text

def clean_text_context(text):
    text = re.sub(text_remove,' ',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def fun(num): 
    if num > df['Move'].mean() + df['Move'].var() * 0.7: 
        return "Buy"
    elif num < df['Move'].mean() - df['Move'].var() * 0.7: 
        return "Sell"
    else: 
        return "Unchanged"
    
def fun2(num): 
    if num > 0: 
        return [1]
    else : 
        return [0]
    
    
def Gentw(path, files):
    tw = pd.DataFrame()
    print('Gen tw')
    carry = str()
    day = datetime.datetime(2013, 12, 31)
    t = datetime.datetime(2013, 12, 31, 9, 30, 0)
    for file in files:
        all_data = [json.loads(line) for line in open(path+"/"+file, 'r')]
        newstr = str()
        t = t + datetime.timedelta(days=1)
        for each_dictionary in all_data:
            text = each_dictionary['text']
            #ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(each_dictionary['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
            cur_t = int(time.strftime('%Y%m%d%H%M%S', time.strptime(each_dictionary['created_at'],'%a %b %d %H:%M:%S +0000 %Y')))
            if cur_t < int(t.strftime('%Y%m%d%H%M%S')):
                carry = carry + text
            else :
                newstr = newstr + text
        day = day + datetime.timedelta(days=1)
        tw_dict = {
                "timestamp": day.strftime("%Y%m%d"),
                "text": carry
        }    
        tweet_json = pd.DataFrame(tw_dict,index = [0])
        #tweet_json['text_tokenized'] = tweet_json.loc[:, 'text'].apply(jieba_tokenizer)
        #print(tweet_json)
        tw = pd.concat([tw, tweet_json]).reset_index(drop=True)
        tw = tw.drop_duplicates()
        carry = newstr
    return tw

def Gendf(tw1, tw2, stock1, stock2, id1, id2): 
    print('Gen df')
    #tw1['text'] = tw1.text.apply(lambda x: clean_text(x))
    #tw2['text'] = tw2.text.apply(lambda x: clean_text(x))
    print('Clean Text')

    df = pd.merge(tw1, tw2[['timestamp', 'text']], on = 'timestamp', how = 'left')

    st1 = pd.read_csv(pricepath + stock1 + '.csv')
    st2 = pd.read_csv(pricepath + stock2 + '.csv')
    PriceGap = pd.DataFrame()
    PriceGap2 = pd.DataFrame()
    PriceGap['timestamp'] = st1['Date'].apply(lambda x : x.replace('-', ''))
    PriceGap2['timestamp'] = st2['Date'].apply(lambda x : x.replace('-', ''))
    PriceGap['gap1'] = (st1['Close'] - st1['Open']) / st1['Open']
    PriceGap2['gap2'] =  (st2['Close'] - st2['Open']) / st2['Open']
    PriceGap = pd.merge(PriceGap, PriceGap2[['timestamp', 'gap2']], on = 'timestamp', how = 'left')
    
    PriceGap['Move'] = PriceGap['gap1'] - PriceGap['gap2']

 
    PriceGap['Move'] = PriceGap['Move'].shift(-1)

    df = pd.merge(df, PriceGap[['timestamp', 'Move']], on = 'timestamp', how = 'left')
    df = df.dropna()
    df['Move'] = df['Move'].apply(fun2)
    df['timestamp'] = df['timestamp'].astype(int)
    
    
    return df

def GenProfit(tw1, tw2, stock1, stock2, id1, id2): 
    print('Gen df')

    df = pd.merge(tw1, tw2[['timestamp', 'text']], on = 'timestamp', how = 'left')

    st1 = pd.read_csv(pricepath + stock1 + '.csv')
    st2 = pd.read_csv(pricepath + stock2 + '.csv')
    PriceGap = pd.DataFrame()
    PriceGap2 = pd.DataFrame()
    PriceGap['timestamp'] = st1['Date'].apply(lambda x : x.replace('-', ''))
    PriceGap2['timestamp'] = st2['Date'].apply(lambda x : x.replace('-', ''))
    PriceGap['gap1'] = ((st1['Close'] - st1['Open']) / st1['Open']) * 50000
    PriceGap2['gap2'] =  ((st2['Close'] - st2['Open']) / st2['Open']) * 50000
    PriceGap = pd.merge(PriceGap, PriceGap2[['timestamp', 'gap2']], on = 'timestamp', how = 'left')
    #PriceGap['gap2'] = preprocessing.scale(st2['Close'] - st2['Open'])
    PriceGap['Move'] = PriceGap['gap1'] - PriceGap['gap2']
    PriceGap['dir1'] = PriceGap['gap1'].apply(fun2)
    PriceGap['dir2'] = PriceGap['gap2'].apply(fun2)
    
    
    PriceGap['Move'] = PriceGap['Move'].shift(-1)
    PriceGap['dir1'] = PriceGap['dir1'].shift(-1)
    PriceGap['dir2'] = PriceGap['dir2'].shift(-1)

    df = pd.merge(df, PriceGap[['timestamp', 'Move']], on = 'timestamp', how = 'left')
    df = pd.merge(df, PriceGap[['timestamp', 'dir1']], on = 'timestamp', how = 'left')
    df = pd.merge(df, PriceGap[['timestamp', 'dir2']], on = 'timestamp', how = 'left')
    
    df = df.dropna()
    df['timestamp'] = df['timestamp'].astype(int)
    
    
    return df

def ComputePro(com):
    profit = 0
    plotpro = {}    
    for i in range(len(com)):
        if int(com['predict_labels'][i]) == 1:
            profit += com['label'][i]
            plotpro[com['front_testid'][i]] += com['label'][i]
        else:
            profit -= com['label'][i]
            plotpro[com['front_testid'][i]] -= com['label'][i]
            
    print('Profit : ', profit)
    
    return profit, plotpro

def ComProDir(test, pro):
    profit = 0
    for i in range(len(pro)):
        if (pro['predict_labels'][i] == [1]) & (pro['predict_labels1'][i] == [0]):
            profit += test['label'][i]
        elif (pro['predict_labels'][i] == [0]) & (pro['predict_labels1'][i] == [1]):
            profit -= test['label'][i]
    print (profit)
    
    return profit


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

def deal(df):
    df['text_x'] = df.text_x.apply(lambda x: clean_text(x))
    df['text_y'] = df.text_y.apply(lambda x: clean_text(x))
    df['t'] = df['timestamp']
    df = df[['timestamp', 't', 'text_x', 'text_y', 'Move', 'dir1', 'dir2']]
    df.columns = ['front_testid', 'behind_testid', 'front_features', 'behind_features', 'label', 'labels_index1', 'labels_index2']
    df['front_testid'] = df['front_testid'].astype(int)
    df['behind_testid'] = df['behind_testid'].astype(int)
    return df
