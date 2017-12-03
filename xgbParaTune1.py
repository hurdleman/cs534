#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:34:17 2017

@author: jeffreyhsu
"""

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
#import matplotlib.pylab as plt
#from matplotlib.pylab import rcParams

trainDataNor = pd.read_pickle('trainDataNorSub.pkl')
trainLabel = pd.read_pickle('trainLabelSub.pkl')

#trainDataNorSub1 = pd.DataFrame(trainDataNorSub.iloc[0::10].copy())
#trainLabelSub1 = pd.DataFrame(trainLabelSub.iloc[0::10].copy())

#rcParams['figure.figsize'] = 12, 4

train = pd.concat([trainDataNor, trainLabel], axis=1)
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
#    plt.ylabel('Feature Importance Score')
    
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
 'min_child_weight':list([2,3,4])
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_



param_test2a = {
 'max_depth':list([1,2,3]),
 'min_child_weight':list([3,4,5])
}
gsearch2a = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2a, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2a.fit(train[predictors],train[target])
gsearch2a.grid_scores_, gsearch2a.best_params_, gsearch2a.best_score_


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=1,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)