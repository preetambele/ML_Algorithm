import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from  sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
from joblib import dump,load
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv('E:\\Download\\Resource\\Advertising.csv')
# print(df)
# print(df.info())

# df['Total_Spend']=df['TV']+ df['radio']+df['newspaper']

print(df)
# print(df.info())

# sns.scatterplot(data=df,x='Total_Spend',y='sales')
# sns.regplot(data=df,x='Total_Spend',y='sales')
# plt.show()

# X=df['Total_Spend']
# y=df['sales']

#y=mx+b
#y=B1x +B0
# print(np.polyfit(X,y,deg=1))

# potential_spend=np.linspace(0,500,100)
# potentail_sales=0.04868788 * potential_spend + 4.24302822

# sns.scatterplot(data=df,x='Total_Spend',y='sales')
# plt.plot(potential_spend,potentail_sales)
# plt.show()

# spend=200
# potentail_sale=0.04868788 * spend + 4.24302822
# print(potentail_sale)

# fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
# axes[0].plot(df['TV'],df['sales'],'o')
# axes[0].set_xlabel('TV Spend')
# axes[0].set_ylabel('Sales')
#
# axes[1].plot(df['radio'],df['sales'],'o')
# axes[1].set_xlabel('Radio Spend')
# axes[1].set_ylabel('sales')
#
# axes[2].plot(df['newspaper'],df['sales'],'o')
# axes[2].set_xlabel('Newspaper sales')
# axes[2].set_ylabel('Sales')
# sns.pairplot(df)
# plt.show()

# X=df.drop('sales',axis=1)
# y=df['sales']
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=101)
# model=LinearRegression()
# model.fit(X_train,y_train)
#
# test_prediction=model.predict(X_test)

# print(mean_absolute_error(y_test,test_prediction))
# print(mean_squared_error(y_test,test_prediction))
# print(root_mean_squared_error(y_test,test_prediction))

# test_residual=y_test - test_prediction
# print(test_residual)

# sns.scatterplot(x=y_test,y=test_residual)
# plt.axhline(y=0,color='r',ls='--')
# sns.displot(test_residual,bins=10,kde=True)
# plt.show()

# final_model=LinearRegression()
# final_model.fit(X,y)
# print(final_model.coef_)
# y_hat=final_model.predict(X)

# print(y_hat)
# fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10,4))
#
# axes[0].plot(df['TV'],df['sales'],'o')
# axes[0].plot(df['TV'],y_hat,'o',color='r')
# axes[0].set_title('TV spend')
# axes[0].set_ylabel('Sales')
#
# axes[1].plot(df['radio'],df['sales'],'o')
# axes[1].plot(df['radio'],y_hat,'o',color='r')
# axes[1].set_title('Radio spend')
# axes[1].set_ylabel('Sales')
#
# axes[2].plot(df['newspaper'],df['sales'],'o')
# axes[2].plot(df['newspaper'],y_hat,'o',color='r')
# axes[2].set_title('Newspaper spend')
# axes[2].set_ylabel('Sales')
#
# plt.show()

# dump(final_model,'final_sales_model.joblib')
# loaded_model=load('final_sales_model.joblib')
# loaded_model.coef_
# print(X.shape)

# campaign=[[149,22,12]]
#
# print(loaded_model.predict(campaign))


#013 Polynomial Regression - Creating Polynomial Features

X=df.drop('sales',axis=1)
y=df['sales']

# polynomial_converter=PolynomialFeatures(degree=2, include_bias=False)
#
# polynomial_converter.fit(X)
# poly_feature=polynomial_converter.transform(X)
#
# X_train,X_test,y_train,y_test=train_test_split(poly_feature,y,test_size=0.30,random_state=101)

# model=LinearRegression()
# model.fit(X_train,y_train)
#
# test_prediction=model.predict(X_test)

# MAE=mean_absolute_error(y_test,test_prediction)
# print(MAE)
# MSE=mean_squared_error(y_test,test_prediction)
# print(MSE)
# RMSE=root_mean_squared_error(y_test,test_prediction)
# print(RMSE)

train_rmse_errors=[]
test_rmse_errors=[]

for d in range(1,10):
    poly_converter=PolynomialFeatures(degree=d,include_bias=False)
    poly_feature=poly_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(poly_feature, y, test_size=0.30, random_state=101)

    model=LinearRegression()
    model.fit(X_train,y_train)

    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    train_rmse=np.sqrt(mean_squared_error(y_train,train_predict))
    test_rmse=np.sqrt(mean_squared_error(y_test,test_predict))

    train_rmse_errors.append(train_rmse)
    test_rmse_errors.append(test_rmse)

# print(train_rmse_errors)
# print(test_rmse_errors)

# plt.plot(range(1,6),train_rmse_errors[:5],label='TRAIN RMSE')
# plt.plot(range(1,6),test_rmse_errors[:5],label='TEST RMSE')
# plt.xlabel('Degree Of Poly')
# plt.ylabel('RMSE')
# plt.legend()
# plt.show()

final_poly_converter=PolynomialFeatures(degree=3,include_bias=False)
final_model=LinearRegression()
full_converter_X=final_poly_converter.fit_transform(X)
final_model.fit(full_converter_X,y)
dump(final_model,'final_poly_model.joblib')
dump(final_poly_converter,'final_converter.joblib')

loaded_converter=load('final_converter.joblib')
loaded_model=load('final_poly_model.joblib')

campaign=[[149,22,122]]

tranformed_data=loaded_converter.fit_transform(campaign)

print(loaded_model.predict(tranformed_data))




