import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train.head()
test.head()
len(train.columns)
len(test.columns)

#Number of missing value in each case
labels = []
values = []
for col in train.columns:
    labels.append(col)
    values.append(train[col].isnull().sum())
    print(col, values[-1])

##Train missing values
# ID 0
# timestamp 0
# Stock_ID 0
# Volume 0
# Three_Day_Moving_Average 448
# Five_Day_Moving_Average 902
# Ten_Day_Moving_Average 2047
# Twenty_Day_Moving_Average 4404
# True_Range 0
# Average_True_Range 2985
# Positive_Directional_Movement 224
# Negative_Directional_Movement 224
# Outcome 0


labels = []
values = []
for col in test.columns:
    labels.append(col)
    values.append(test[col].isnull().sum())
    print(col, values[-1])
##Test missing values
# ID 0
# timestamp 0
# Stock_ID 0
# Volume 0
# Three_Day_Moving_Average 54
# Five_Day_Moving_Average 127
# Ten_Day_Moving_Average 280
# Twenty_Day_Moving_Average 587
# True_Range 0
# Average_True_Range 401
# Positive_Directional_Movement 100
# Negative_Directional_Movement 61

train['Outcome'].value_counts()
#0    384456
#1    318283
#Name: Outcome, dtype: int64

train.shape
#(702739, 13)

test.shape
#(101946, 12)

test['Outcome']='NA'
train_test = train.append(test)
train_test.describe()
train_test.head(5)


labels = []
values = []
for col in train_test.columns:
    labels.append(col)
    values.append(train_test[col].isnull().sum())
    print(col, values[-1])
# ID 0
# timestamp 0
# Stock_ID 0
# Volume 0
# Three_Day_Moving_Average 502
# Five_Day_Moving_Average 1029
# Ten_Day_Moving_Average 2327
# Twenty_Day_Moving_Average 4991
# True_Range 0
# Average_True_Range 3386
# Positive_Directional_Movement 324
# Negative_Directional_Movement 285
# Outcome 0


#$$$$$$$#Mising values treat
train_test[(train_test['Three_Day_Moving_Average'].isnull())&(train_test['Five_Day_Moving_Average'].isnull())]['ID']
set(train_test[(train_test['Three_Day_Moving_Average'].isnull())&(train_test['Five_Day_Moving_Average'].isnull())]['ID']) - set(train_test[(train_test['Three_Day_Moving_Average'].isnull())]['ID'])
#set() #So Five_Day_Moving_Average is subset of Three_Day_Moving_Average

set(train_test[(train_test['Five_Day_Moving_Average'].isnull())&(train_test['Ten_Day_Moving_Average'].isnull())]['ID']) - set(train_test[(train_test['Five_Day_Moving_Average'].isnull())]['ID'])
#set()

set(train_test[(train_test['Ten_Day_Moving_Average'].isnull())&(train_test['Twenty_Day_Moving_Average'].isnull())]['ID']) - set(train_test[(train_test['Ten_Day_Moving_Average'].isnull())]['ID'])
#set()


#$#Need to find closing price for the day

#Strategy: Missing values impute for Three_Day_Moving_Average and use it to impute Five_Day_Moving_Average. Use them (Three_Day_Moving_Average and Five_Day_Moving_Average) to find Closing price of each day. Impute missing values of the Ten_Day_Moving_Average and Twenty_Day_Moving_Average using Closing Price.
#1*#Missing in Three_Day_Moving_Average of nth day = {[n-1]th day Three_Day_Moving_Average +[n+1]th day Three_Day_Moving_Average}/2
#2*#Missing in Five_Day_Moving_Average of nth day = {([n-2]th Three_Day_Moving_Average)+([n]th Three_Day_Moving_Average)}/2
#3*#Closing Price of n-th day = 3*(Three_Day_Moving_Average of [n+1]th day) - {5*(Five_Day_Moving_Average of [n+3]th day) - 3*(Three_Day_Moving_Average of [n+3]th day)}

#1*
for i in list(train_test.index)[1:-1]:
	if math.isnan(train_test.iloc[i,:]['Three_Day_Moving_Average']) & math.isnan(train_test.iloc[i-1,:]['Three_Day_Moving_Average']) & math.isnan(train_test.iloc[i+1,:]['Three_Day_Moving_Average']):
		print(i)
#Outputs nothing

for i in list(train_test.index)[1:-1]:
	if math.isnan(train_test.iloc[i,:]['Three_Day_Moving_Average'])==True & math.isnan(train_test.iloc[i-1,:]['Three_Day_Moving_Average'])==False & math.isnan(train_test.iloc[i+1,:]['Three_Day_Moving_Average'])==False:
		av=(train_test.iloc[i-1,:]['Three_Day_Moving_Average']+train_test.iloc[i+1,:]['Three_Day_Moving_Average'])/2
		train_test.set_value(index=i,col='Three_Day_Moving_Average',value=av)
	if math.isnan(train_test.iloc[i,:]['Three_Day_Moving_Average'])==True & math.isnan(train_test.iloc[i-1,:]['Three_Day_Moving_Average'])==True & math.isnan(train_test.iloc[i+1,:]['Three_Day_Moving_Average'])==False:
		train_test.set_value(index=i,col='Three_Day_Moving_Average',value=train_test.iloc[i+1,:]['Three_Day_Moving_Average'])
	if math.isnan(train_test.iloc[i,:]['Three_Day_Moving_Average'])==True & math.isnan(train_test.iloc[i-1,:]['Three_Day_Moving_Average'])==False & math.isnan(train_test.iloc[i+1,:]['Three_Day_Moving_Average'])==True:
		train_test.set_value(index=i,col='Three_Day_Moving_Average',value=train_test.iloc[i-1,:]['Three_Day_Moving_Average'])

Three_Day_MA=list(train_test['Three_Day_Moving_Average'])
for i in range(1,len(Three_Day_MA)-1):
	if math.isnan(Three_Day_MA[i])==True & math.isnan(Three_Day_MA[i-1])==False & math.isnan(Three_Day_MA[i+1])==False:
		Three_Day_MA[i]=(Three_Day_MA[i-1]+Three_Day_MA[i+1])/2
	if math.isnan(Three_Day_MA[i])==True & math.isnan(Three_Day_MA[i-1])==True & math.isnan(Three_Day_MA[i+1])==False:
		Three_Day_MA[i]=Three_Day_MA[i+1]
	if math.isnan(Three_Day_MA[i])==True & math.isnan(Three_Day_MA[i-1])==False & math.isnan(Three_Day_MA[i+1])==True:
		Three_Day_MA[i]=Three_Day_MA[i-1]
train_test['Three_Day_Moving_Average']=Three_Day_MA

for i in list(train_test.index)[1:-1]:
	if math.isnan(train_test.iloc[i,:]['Three_Day_Moving_Average'])==True & math.isnan(train_test.iloc[i-1,:]['Three_Day_Moving_Average'])==False & math.isnan(train_test.iloc[i+1,:]['Three_Day_Moving_Average'])==False:
		av=(train_test.iloc[i-1,:]['Three_Day_Moving_Average']+train_test.iloc[i+1,:]['Three_Day_Moving_Average'])/2
		train_test.set_value(index=i,col='Three_Day_Moving_Average',value=av)
	if math.isnan(train_test.iloc[i,:]['Three_Day_Moving_Average'])==True & math.isnan(train_test.iloc[i-1,:]['Three_Day_Moving_Average'])==True & math.isnan(train_test.iloc[i+1,:]['Three_Day_Moving_Average'])==False:
		train_test.set_value(index=i,col='Three_Day_Moving_Average',value=train_test.iloc[i+1,:]['Three_Day_Moving_Average'])
	if math.isnan(train_test.iloc[i,:]['Three_Day_Moving_Average'])==True & math.isnan(train_test.iloc[i-1,:]['Three_Day_Moving_Average'])==False & math.isnan(train_test.iloc[i+1,:]['Three_Day_Moving_Average'])==True:
		train_test.set_value(index=i,col='Three_Day_Moving_Average',value=train_test.iloc[i-1,:]['Three_Day_Moving_Average'])

##np.count_nonzero(np.isnan(Three_Day_MA))
		
#2*
for i in list(train_test.index)[2:]:
	if math.isnan(train_test.iloc[i,:]['Five_Day_Moving_Average']):
		train_test.set_value(index=i,col='Five_Day_Moving_Average',value=(train_test.iloc[i-2,:]['Three_Day_Moving_Average']+train_test.iloc[i,:]['Three_Day_Moving_Average'])/2)

#3*
train_test=train_test.fillna(0)

cp=[]
for i in list(train_test.index):
	cp.append(3*train_test.iloc[i+1,:]['Three_Day_Moving_Average']-5*train_test.iloc[i+3,:]['Five_Day_Moving_Average']+3*train_test.iloc[i+3,:]['Three_Day_Moving_Average'])

train_test["Closing_Price"]=cp


response=[]
for i in range(len(cp)-1):
	if cp[i+1]>cp[i]:
		response.append(1)
	else:
		response.append(0)
resonse.append(1)
response_actual_train=list(train.iloc['Outcome'])
response_obtained_train=response[0:702738]
response_obtained_test=response[702739:]

cm=confusion_matrix(response_actual_train, response_obtained_train)
print("Train CM")
print(cm)

#train_test.iloc[i,:]["Outcome"]
submit = pd.DataFrame({'ID': test['ID'], 'Outcome': response_obtained_test})
submit.to_csv("Normal_no_treatment_output.csv", index=False)



$#Need today high
$#Need today low
$Outliers remove
$Predict Closing price of today given the features 
$Time Series check to predict the next close value: if more then today then 1 else 0
$LSTM to predict the next close value: if more then today then 1 else 0



########Simple Classification Make
# import pandas as pd
# import numpy as np
# from sklearn import preprocessing, model_selection, metrics, ensemble
# import xgboost as xgb
# train=pd.read_csv("train.csv")
# test=pd.read_csv("test.csv")
# train_test = train.append(test)
# features = np.setdiff1d(train_test.columns, ['ID', 'Outcome'])

# X_train=train_test[0:len(train.index)]
# X_test=train_test[len(train.index):len(train_test.index)]

# params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
#                 "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
#                 "min_child_weight": 1, "num_class": 2,
#                 "seed": 2016, "tree_method": "exact"}



# dtrain = xgb.DMatrix(X_train[features], X_train['Outcome'], missing=np.nan)
# dtest = xgb.DMatrix(X_test[features], missing=np.nan)

# nrounds = 260
# watchlist = [(dtrain, 'train')]
# bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
# test_preds = bst.predict(dtest)

# submit = pd.DataFrame({'ID': test['ID'], 'Outcome': test_preds})
# submit.to_csv("XGB_no_treatment_output.csv", index=False)

