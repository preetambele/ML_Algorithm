import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.linear_model import Ridge
from  sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
from sklearn.linear_model import RidgeCV
# from sklearn.metrics import SCORERS
from  sklearn.linear_model import  LassoCV
from sklearn.linear_model import ElasticNetCV

df=pd.read_csv('E:\\Download\\Resource\\Advertising.csv')

X=df.drop('sales',axis=1)
y=df['sales']

polynomial_converter=PolynomialFeatures(degree=3,include_bias=False)

poly_feature=polynomial_converter.fit_transform(X)

# print(polynomial_feature.shape)

X_train,X_test,y_train,y_test=train_test_split(poly_feature,y,test_size=0.30,random_state=101)

scaler=StandardScaler()

scaler.fit(X_train)

X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

ridge_model=Ridge(alpha=10)
ridge_model.fit(X_train,y_train)

test_prediction=ridge_model.predict(X_test)

# print(mean_squared_error(y_test,test_prediction))
# print(mean_absolute_error(y_test,test_prediction))
# print(root_mean_squared_error(y_test,test_prediction))

ridge_cv_model=RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error')

ridge_cv_model.fit(X_train,y_train)

# print(ridge_cv_model.alpha_)
test_predictions=ridge_cv_model.predict(X_test)

# print(mean_squared_error(y_test,test_predictions))
# print(mean_absolute_error(y_test,test_predictions))
# print(root_mean_squared_error(y_test,test_predictions))

lasso_cv_model=LassoCV(eps=0.1,n_alphas=100,cv=5)
lasso_cv_model.fit(X_train,y_train)

# print(lasso_cv_model.alpha_)
test_predict=lasso_cv_model.predict(X_test)

# print(mean_squared_error(y_test,test_predict))
# print(mean_absolute_error(y_test,test_predict))
# print(root_mean_squared_error(y_test,test_predict))


#025 L1 and L2 Regularization - Elastic Net

elastic_model=ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,.99,1],eps=0.001,n_alphas=100,max_iter=100000)

elastic_model.fit(X_train,y_train)

# print(elastic_model.l1_ratio_)

test_predict1=elastic_model.predict(X_test)