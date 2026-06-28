import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from  sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('E:\\Download\\Resource\\Advertising.csv')

print(df.info())

X=df.drop('sales',axis=1)
y=df['sales']

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
#
# scaler=StandardScaler()
# scaler.fit(X_train)
# x_train=scaler.transform(X_train)
# x_test=scaler.transform(X_test)
#
# model=Ridge(alpha=100)
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
#
# print(mean_squared_error(y_test,y_pred))
#
# model1=Ridge(alpha=1)
# model1.fit(X_train,y_train)
# y_pred1=model1.predict(X_test)
#
# print(mean_squared_error(y_test,y_pred1))


#-------------------------------003 Cross Validation - Test  Validation  Train Split-------------------------------

# X_train,X_other,y_train,y_other=train_test_split(X,y,test_size=0.3,random_state=101)
# X_eval,X_test,y_eval,y_test   =train_test_split(X_other,y_other,test_size=.5,random_state=101)
# scaler=StandardScaler()
# scaler.fit(X_train)
#
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)
# X_eval=scaler.transform(X_eval)
#
# model_one=Ridge(alpha=100)
# model_one.fit(X_train,y_train)
#
# y_eval_pred=model_one.predict(X_eval)
# print(mean_squared_error(y_eval,y_eval_pred))
#
# model_two=Ridge(alpha=1)
# model_two.fit(X_train,y_train)
# new_eval_pred=model_two.predict(X_eval)
# print(mean_squared_error(y_eval,new_eval_pred))
#
# y_final_pred=model_two.predict(X_test)
# print(mean_squared_error(y_test,y_final_pred))

# ----------------------------------------004 Cross Validation - cross_val_score---------------------------------

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
#
# scaler=StandardScaler()
# scaler.fit(X_train)
#
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)
#
# model=Ridge(alpha=100)
#
# scores=cross_val_score(model,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
#
# print(scores)
# print(abs(scores.mean()))
# model1=Ridge(alpha=1)
# score=cross_val_score(model1,X_train,y_train,scoring='neg_mean_squared_error',cv=5)
# print(score)
# print(abs(score.mean()))
#
# model1.fit(X_train,y_train)
#
# y_final_test_predict=model1.predict(X_test)
# print(mean_squared_error(y_test,y_final_test_predict))

#---------------------------------------005 Cross Validation - cross_validate------------------------------

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
#
# scaler=StandardScaler()
# scaler.fit(X_train)
#
# X_train=scaler.transform(X_train)
# X_test=scaler.transform(X_test)
#
# model=Ridge(alpha=100)
#
# scores=cross_validate(model,X_train,y_train,scoring=['neg_mean_absolute_error','neg_mean_squared_error'],cv=10)
# scores=pd.DataFrame(scores)
# print(scores.mean())
#
# model1=Ridge(alpha=1)
# score=cross_validate(model1,X_train,y_train,scoring=['neg_mean_absolute_error','neg_mean_squared_error'],cv=10)
# score=pd.DataFrame(score)
# print(score.mean())
#
# model1.fit(X_train,y_train)
# y_final_predict=model1.predict(X_test)
# print(mean_squared_error(y_test,y_final_predict))

#--------------------------------------------------006 Grid Search----------------------------------------------
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

base_elastic_net_model=ElasticNet()

param_graid={'alpha':[0.1,1,5,10,50,100],'l1_ratio':[0.1,.5,.7,.95,.99,1]}

grid_model=GridSearchCV(estimator=base_elastic_net_model,
                        param_grid=param_graid,
                        scoring='neg_mean_squared_error',cv=5,verbose=1)

grid_model.fit(X_train,y_train)

print(grid_model.best_estimator_)
print(grid_model.best_params_)

y_pred=grid_model.predict(X_test)

print(mean_squared_error(y_test,y_pred))



