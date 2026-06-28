import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import GridSearchCV

pd.set_option('display.width',900)
pd.set_option('display.max_columns',11)

df=pd.read_csv('E:\\Download\\Resource\\09_Support_Vector_Machines\\cement_slump.csv')
print(df.columns)

# sns.heatmap(df.corr(),annot=True)
# plt.show()
X=df.drop('Compressive Strength (28-day)(Mpa)',axis=1)
y=df['Compressive Strength (28-day)(Mpa)']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

base_model=SVR()
base_model.fit(scaled_X_train,y_train)

base_preds=base_model.predict(scaled_X_test)


print(mean_absolute_error(y_test,base_preds))
print(np.sqrt(mean_squared_error(y_test,base_preds)))

param_grid={'C':[0.001,0.01,0.1,0.5,1],
            'kernel':['linear','rdf','poly'],
            'gamma':['scale','auto'],
            'degree':[2,3,4],
            'epsilon':[0,0.01,0.1,0.5,1,2]}

svr=SVR()

grid=GridSearchCV(svr,param_grid)
grid.fit(scaled_X_train,y_train)

# print(grid.best_params_)

grid_preds=grid.predict(scaled_X_test)

print(mean_absolute_error(y_test,grid_preds))
print(np.sqrt(mean_squared_error(y_test,grid_preds)))



