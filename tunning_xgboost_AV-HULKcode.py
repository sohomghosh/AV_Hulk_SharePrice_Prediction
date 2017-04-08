import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, metrics, ensemble
import xgboost as xgb

from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train_test = train.append(test)
features = np.setdiff1d(train_test.columns, ['ID', 'Outcome'])

X_train=train_test[0:len(train.index)]
X_test=train_test[len(train.index):len(train_test.index)]

#dtrain = xgb.DMatrix(X_train[features], X_train['Outcome'], missing=np.nan)
#dtest = xgb.DMatrix(X_test[features], missing=np.nan)

#params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
#                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
#                "min_child_weight": 1, "num_class": 2,
#                "seed": 2016, "tree_method": "exact"}
# nrounds = 260
# watchlist = [(dtrain, 'train')]
# bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
# test_preds = bst.predict(dtest)



# submit = pd.DataFrame({'ID': test['ID'], 'Outcome': test_preds})
# submit.to_csv("XGB_no_treatment_output1.csv", index=False)




cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1) 
optimized_GBM.fit(X_train[features], X_train['Outcome'])
GridSearchCV(cv=5, error_score='raise',
       estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.8),
       fit_params={}, iid=True, n_jobs=-1,
       param_grid={'min_child_weight': [1, 3, 5], 'max_depth': [3, 5, 7]},
       pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)
optimized_GBM.grid_scores_


cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth': 3, 'min_child_weight': 1}


optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM.fit(X_train[features], X_train['Outcome'])
GridSearchCV(cv=5, error_score='raise',
       estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1),
       fit_params={}, iid=True, n_jobs=-1,
       param_grid={'subsample': [0.7, 0.8, 0.9], 'learning_rate': [0.1, 0.01]},
       pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)
optimized_GBM.grid_scores_

xgdmat = xgb.DMatrix(X_train[features], X_train['Outcome'], missing=np.nan)

# Grid Search CV optimized settings from before use here
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 


cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error
cv_xgb.tail(5)


#Using best setting parameters from before
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432)


#%matplotlib inline
import seaborn as sns
sns.set(font_scale = 1.5)
xgb.plot_importance(final_gb)
importances = final_gb.get_fscore()
importances
importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
#change figure size (8,8) if needed
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')

testdmat = xgb.DMatrix(X_test)
from sklearn.metrics import accuracy_score
y_pred = final_gb.predict(testdmat) # Predict using our testdmat
y_pred

y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
y_pred

accuracy_score(y_pred, y_test), 1-accuracy_score(y_pred, y_test)