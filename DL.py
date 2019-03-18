# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:03:19 2019

@author: Administrator
"""

import pandas as pd
import re
import gensim
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Embedding,SimpleRNN
from keras.layers import MaxPool2D,Concatenate,Dense,Dropout
from keras import models,layers
from keras.layers import Input,Reshape,Conv2D,Concatenate,Flatten, Dropout,Dense
from keras.models import Model

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

data = pd.read_csv('总数据.csv',encoding='ISO-8859-1')
review = data['text']
label = data['label']
def review_to_words(raw_review):
    review_re = re.sub('[^a-zA-Z]',' ',raw_review)
    review_lower = review_re.lower()
    review_split = review_lower.split()
    Stem = WordNetLemmatizer()
    review_stem = [Stem.lemmatize(word) for word in review_split]
    stops = set(stopwords.words('english'))
    review_afterstop = [w for w in review_stem if not w in stops]
    return(' '.join(review_afterstop))
all_review = [] 
for i in range(len(review)):
    if ((i+1)%5000 == 0):
        print('已处理第%d/%d条数据' % ((i+1),len(review)))
    all_review.append(review_to_words(review[i]))

words=[]
for raw_sentence in all_review:
    words.append(raw_sentence.split())

word_to_ix = {}
for sent  in words:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
max_features = len(word_to_ix)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(input_train))
train_X = tokenizer.texts_to_sequences(input_train)
test_X = tokenizer.texts_to_sequences(input_test)
maxlen = 15
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

model.summary()
history = model.fit(train_X,y_train,epochs=10,batch_size=128,validation_split=0.2)
pred_y_test = model.predict(test_X, batch_size=1024, verbose=1)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))

from keras.layers import LSTM

model = Sequential() 
model.add(Embedding(max_features, 32)) 
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
history = model.fit(train_X,y_train,epochs=10,batch_size=128,validation_split=0.2)

pred_y_test = model.predict(test_X, batch_size=1024, verbose=1)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))


from keras.layers import MaxPool2D,Concatenate,Dense,Dropout
embed_size = 300
maxlen = 15

def Text_CNN():

    filter_sizes = [2,3] #不同size的filter
    num_filters = 20 #同一size filter的数量

    inp = Input(shape=(maxlen,))

    #embedding层
    x = Embedding(max_features,embed_size)(inp) #non-static模式embedding
    x = Reshape((maxlen,embed_size,1))(x)

    maxpoolList=[]

    #卷积层和pooling层
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters,kernel_size=(filter_sizes[i],embed_size),
                      kernel_initializer='he_normal',activation='elu')(x)
        maxpoolList.append(MaxPool2D(pool_size=(maxlen-filter_sizes[i]+1,1))(conv))

    #连接pooling层输出的结果
    z = Concatenate(axis=1)(maxpoolList)
    z = Flatten()(z)

    #dropout正则化
    z = Dropout(0.5)(z)

    #全连接层
    y = Dense(1,activation="sigmoid")(z)


    model = Model(inputs=inp,outputs=y)

    #定义损失函数和优化器
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model
    
model = Text_CNN()
model.summary()
history = model.fit(train_X,y_train,epochs=10,batch_size=128,validation_split=0.2)

pred_y_test = model.predict(test_X, batch_size=1024, verbose=1)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))
    
    