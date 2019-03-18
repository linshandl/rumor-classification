# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:54:20 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer as TFV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


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
for i in range(len(review)):a
    if ((i+1)%5000 == 0):
        print('已处理第%d/%d条数据' % ((i+1),len(review)))
    all_review.append(review_to_words(review[i]))
    
tfv = TFV(analyzer='word',min_df=3)  #词频低于3则忽略
all_data = tfv.fit_transform(all_review) 

x_train_and_val,x_test,y_train_and_val,y_test = train_test_split(all_data,label,random_state=0) #test_size 忘了写0.3
print('train_and_validation:{}\ntest:{}'.format(x_train_and_val.shape[0],x_test.shape[0]))

NB_grid_values = {'alpha':[1,5,10]}
NB_grid_search = GridSearchCV(MNB(fit_prior=True,class_prior=None),NB_grid_values,cv=5)
NB_grid_search.fit(x_train_and_val,y_train_and_val)
print('NB best params:{}'.format(NB_grid_search.best_params_))
print('NB score on train set:{:.5f}'.format(NB_grid_search.best_score_))
print('NB score on test set:{:.5f}'.format(NB_grid_search.score(x_test,y_test)))


NB = MNB(alpha=1.0)
NB.fit(x_train_and_val,y_train_and_val)
pred_y_test = NB.predict(x_test)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))

LR_grid_values = {'C':[1,20,50,80,100]}
LR_grid_search = GridSearchCV(LR(penalty='l2',random_state=0),LR_grid_values,cv=5)
LR_grid_search.fit(x_train_and_val,y_train_and_val)
print('LR best params:{}'.format(LR_grid_search.best_params_))
print('LRscore on train set:{:.5f}'.format(LR_grid_search.best_score_))
print('LRscore on test set:{:.5f}'.format(LR_grid_search.score(x_test,y_test)))

LR = LR(C=50)
LR.fit(x_train_and_val,y_train_and_val)
pred_y_test = LR.predict(x_test)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))


RF_grid_values = {'n_estimators':[50,80,100]}
RF_grid_search = GridSearchCV(RF(random_state=0),RF_grid_values,cv=5)
RF_grid_search.fit(x_train_and_val,y_train_and_val)
print('RF best params:{}'.format(RF_grid_search.best_params_))
print('RF score on train set:{:.5f}'.format(RF_grid_search.best_score_))
print('RF score on test set:{:.5f}'.format(RF_grid_search.score(x_test,y_test)))

RF = RF(n_estimators=100)
RF.fit(x_train_and_val,y_train_and_val)
pred_y_test = RF.predict(x_test)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))


SVM_grid_values = {'gamma':[0.1,0.3,0.5],'C':[0.1,1,10]} #gamma是选择RBF函数作为kernel后
SVM_grid_search = GridSearchCV(SVC(random_state=0),SVM_grid_values,cv=5)
SVM_grid_search.fit(x_train_and_val,y_train_and_val)
print('SVM best params:{}'.format(SVM_grid_search.best_params_))
print('SVM score on train set:{:.5f}'.format(SVM_grid_search.best_score_))
print('SVM score on test set:{:.5f}'.format(SVM_grid_search.score(x_test,y_test)))

SVM = SVC(gamma=0.5,C=10)
SVM.fit(x_train_and_val,y_train_and_val)
pred_y_test = SVM.predict(x_test)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))

y_gbc = gbc.predict(x_train_and_val)
y_gbc1 = gbc.predict(x_test)
acc_train = gbc.score(x_train_and_val, y_train_and_val)
acc_test = gbc.score(x_test, y_test)
print("GBDT score on train set:{:.5f}" .format(acc_train))
print("GBDT score on test set:{:.5f}" .format(acc_test))

GBDT_grid_values = {'n_estimators':range(50,201,50)} #gamma是选择RBF函数作为kernel后
GBDT_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=0),GBDT_grid_values,cv=5)
GBDT_grid_search.fit(x_train_and_val,y_train_and_val)
print('GBDT best params:{}'.format(GBDT_grid_search.best_params_))
print('GBDT score on train set:{:.5f}'.format(GBDT_grid_search.best_score_))
print('GBDT score on test set:{:.5f}'.format(GBDT_grid_search.score(x_test,y_test)))

GBDT_grid_values = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)} 
GBDT_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=0),GBDT_grid_values,cv=5)
GBDT_grid_search.fit(x_train_and_val,y_train_and_val)
print('GBDT best params:{}'.format(GBDT_grid_search.best_params_))
print('GBDT score on train set:{:.5f}'.format(GBDT_grid_search.best_score_))
print('GBDT score on test set:{:.5f}'.format(GBDT_grid_search.score(x_test,y_test)))

GBDT_grid_values = {'n_estimators':[250],'max_depth':[13], 'min_samples_split':[100]} 
GBDT_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=0),GBDT_grid_values,cv=5)
GBDT_grid_search.fit(x_train_and_val,y_train_and_val)
print('GBDT best params:{}'.format(GBDT_grid_search.best_params_))
print('GBDT score on train set:{:.5f}'.format(GBDT_grid_search.best_score_))
print('GBDT score on test set:{:.5f}'.format(GBDT_grid_search.score(x_test,y_test)))

GBDT_grid_values = {'n_estimators':range(100,301,50),'max_depth':[13], 'min_samples_split':[100],'learning_rate':[0.01,0.05,0.1,0.5,1]} 
GBDT_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=0),GBDT_grid_values,cv=5)
GBDT_grid_search.fit(x_train_and_val,y_train_and_val)
print('GBDT best params:{}'.format(GBDT_grid_search.best_params_))
print('GBDT score on train set:{:.5f}'.format(GBDT_grid_search.best_score_))
print('GBDT score on test set:{:.5f}'.format(GBDT_grid_search.score(x_test,y_test)))

GBC = GradientBoostingClassifier(n_estimators=250,max_depth=13,min_samples_split=100,learning_rate=0.5)
GBC.fit(x_train_and_val,y_train_and_val)
pred_y_test = GBC.predict(x_test)
print("recall score: {}".format(metrics.recall_score(y_test,(pred_y_test>0.5))))
print("accuracy score: {}".format(metrics.accuracy_score(y_test,(pred_y_test>0.5))))
print("F1 score: {}".format(metrics.f1_score(y_test,(pred_y_test>0.5))))



