#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 00:43:37 2017

@author: jeffreyhsu
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

trainDataNorSub = pd.read_pickle('trainDataNorSub.pkl')
trainLabelSub = pd.read_pickle('trainLabelSub.pkl')

#trainDataNorSub1 = pd.DataFrame(trainDataNorSub.iloc[0::10].copy())
#trainLabelSub1 = pd.DataFrame(trainLabelSub.iloc[0::10].copy())

rcParams['figure.figsize'] = 12, 4

train = pd.concat([trainDataNorSub, trainLabelSub], axis=1)
target = 0
IDcol = 'ID'

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report :
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
            
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Impxortances')
    plt.ylabel('Feature Importance Score')
    
    #Choose all predictors except target & IDcols

#%% Step 1: Fix learning rate and number of estimators for tuning tree-based parameters
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)

param_test1 = {
 'max_depth':list(range(3,13,2)),
 'min_child_weight':list(range(1,7,2))
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


param_test2 = {
 'max_depth':list([2,3,4]),
 'min_child_weight':list([1,2,3])
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_



#param_test2a = {
# 'max_depth':list([1,2,3]),
# 'min_child_weight':list([3,4,5])
#}
#gsearch2a = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
# min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test2a, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch2a.fit(train[predictors],train[target])
#gsearch2a.grid_scores_, gsearch2a.best_params_, gsearch2a.best_score_


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)

#%% Step 2: Tune max_depth and min_child_weight
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

modelfit(gsearch3.best_estimator_, train, predictors)

xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb2, train, predictors)


#%%Step 4: Tune subsample and colsample_bytree
param_test4 = {
 'subsample':[i/10.0 for i in range(6,11)],
 'colsample_bytree':[i/10.0 for i in range(3,10,2)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=3,
 min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=11,
 min_child_weight=1,
 gamma=0.2,
 subsample=1,
 colsample_bytree=0.5,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, train, predictors)

#param_test5 = {
# 'subsample':[i/100.0 for i in range(80,101,4)],
# 'colsample_bytree':[i/100.0 for i in range(80,101,4)]
#}
#gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=11,
# min_child_weight=1, gamma=0.2, subsample=0.7, colsample_bytree=0.6,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch5.fit(train[predictors],train[target])
#gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

#%% Step 5: Tuning Regularization Parameters

#param_test6 = {
# 'reg_lambda':[0.1, 1, 100, 10000]
#}
#gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=11,
# min_child_weight=1, gamma=0.2, subsample=1, colsample_bytree=0.88,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch6.fit(train[predictors],train[target])
#gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
#
#
#param_test7 = {
# 'reg_lambda':[5000, 10000, 50000, 100000, 500000]
#}
#gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=11,
# min_child_weight=1, gamma=0.2, subsample=0.75, colsample_bytree=0.65,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch7.fit(train[predictors],train[target])
#gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_

param_test8 = {
 'reg_alpha':[0.01, 0.1, 1, 10, 100]
}
gsearch8 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=11,
 min_child_weight=1, gamma=0.2, subsample=0.75, colsample_bytree=0.65,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test8, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch8.fit(train[predictors],train[target])
gsearch8.grid_scores_, gsearch8.best_params_, gsearch8.best_score_


param_test9 = {
 'reg_alpha':[500, 750, 1000, 1250, 1500]
}
gsearch9 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=11,
 min_child_weight=1, gamma=0.2, subsample=0.75, colsample_bytree=0.65,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test9, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch9.fit(train[predictors],train[target])
gsearch9.grid_scores_, gsearch9.best_params_, gsearch9.best_score_

param_test10 = {
 'reg_alpha':[200, 350, 500, 650, 800]
}
gsearch10 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=11,
 min_child_weight=1, gamma=0.2, subsample=0.75, colsample_bytree=0.65,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test10, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch10.fit(train[predictors],train[target])
gsearch10.grid_scores_, gsearch10.best_params_, gsearch10.best_score_


param_test11 = {
 'reg_alpha':[110, 130, 150, 170, 200]
}
gsearch11 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=11,
 min_child_weight=1, gamma=0.2, subsample=0.75, colsample_bytree=0.65,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test11, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch11.fit(train[predictors],train[target])
gsearch11.grid_scores_, gsearch11.best_params_, gsearch11.best_score_


xgb4 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.2,
 subsample=1,
 colsample_bytree=0.5,
 reg_alpha=170,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, train, predictors)

#%%Step 6: Reducing Learning Rate
xgb5 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.2,
 subsample=1,
 colsample_bytree=0.5,
 reg_alpha=170,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb5, train, predictors)