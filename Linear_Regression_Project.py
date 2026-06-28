import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error

df=pd.read_csv('E:\\Download\\Resource\\AMES_Final_DF.csv')

print(df.info())

X=df.drop('SalePrice',axis=1)
y=df['SalePrice']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=101)

scaler=StandardScaler()
scaler.fit(X_train)

scaled_X_train=scaler.transform(X_train)
scaled_X_test=scaler.transform(X_test)


base_elastic_model=ElasticNet(max_iter=100000)

param_grid={'alpha':[0.1,1,5,10,50,100],'l1_ratio':[.1,.5,.7,95,.99,1]}

grid_model=GridSearchCV(estimator=base_elastic_model,
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error',cv=5,verbose=1)


grid_model.fit(scaled_X_train,y_train)

print(grid_model.best_params_)
print(grid_model.best_estimator_)

y_pred=grid_model.predict(scaled_X_test)

print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))

print(np.mean(df['SalePrice']))

