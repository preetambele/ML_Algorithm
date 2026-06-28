import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor

df=pd.read_csv('E:\\Download\\Resource\\rock_density_xray.csv')

# print(df)

df.columns=['Signal','Density']
print(df)

# sns.scatterplot(x='Signal',y='Density',data=df)
# plt.show()

X=df['Signal'].values.reshape(-1,1)
y=df['Density']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=101)
lr_model=LinearRegression()
lr_model.fit(X_train,y_train)

lr_pred=lr_model.predict((X_test))

# print(mean_absolute_error(y_test,lr_pred))
# print(np.sqrt(mean_squared_error(y_test,lr_pred)))

# sns.scatterplot(x='Signal',y='Density',data=df)
# plt.show()

def run_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    rmse=np.sqrt(mean_squared_error(y_test,pred))
    mae=mean_squared_error(y_test,pred)
    print(f'MAE:{mae}')
    print(f'RMSE:{rmse}')

    signal_range=np.arange(1,100)
    signal_pred=model.predict(signal_range.reshape(-1,1))

    plt.figure(figsize=(7,6))
    sns.scatterplot(x='Signal',y='Density',data=df,color='black')

    plt.plot(signal_range,signal_pred)
    plt.show()

# --------------LinearRegression-----------------------
# model=LinearRegression()
# run_model(model, X_train, X_test, y_train, y_test)

# --------------PolynomialRegression-----------------------
# pipe=make_pipeline(PolynomialFeatures(degree=2),LinearRegression())
# run_model(pipe, X_train, X_test, y_train, y_test)

# --------------KNN-----------------------

# k_vlaues=[1,5,10]
# for n in k_vlaues:
#     model=KNeighborsRegressor(n_neighbors=n)
#     run_model(model, X_train, X_test, y_train, y_test)

# --------------DecisionTree-----------------------
# model=DecisionTreeRegressor()
# run_model(model, X_train, X_test, y_train, y_test)

#--------------------SVM---------------------

# svr=SVR()
# param_grid={'C':[0.01,0.1,1,5,10,100,1000],
#             'gamma':['auto','scale']}
#
# grid=GridSearchCV(svr,param_grid)
# run_model(grid, X_train, X_test, y_train, y_test)


#--------------------RandomForest---------------------
# rfr=RandomForestRegressor(n_estimators=10)
# run_model(rfr, X_train, X_test, y_train, y_test)

#--------------------Booster---------------------
# model=GradientBoostingRegressor()
# run_model(model, X_train, X_test, y_train, y_test)

model=AdaBoostRegressor()
run_model(model, X_train, X_test, y_train, y_test)


