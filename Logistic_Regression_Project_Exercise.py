import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LogisticRegressionCV
from sklearn.metrics import classification_report,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

df=pd.read_csv('E:\\Download\\Resource\\07_Logistic_Regression_Models\\heart.csv')
print(df.info())

# print(df.isnull().sum())
# print(df.describe())
# print(df.corr())
# sns.countplot(x='target',data=df)
# sns.pairplot(df[['age','trestbps', 'chol','thalach','target']],hue='target')
# plt.figure(figsize=(10,5))
# sns.heatmap(df.corr(),annot=True)
# plt.show()

X=df.drop('target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=101)

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

log_model=LogisticRegressionCV()

log_model.fit(scaled_X_train,y_train)

# print(log_model.C_)
# print(log_model.get_params())
# print(log_model.coef_)
coefs = pd.Series(index=X.columns,data=log_model.coef_[0])
coefs=coefs.sort_values()
# print(coefs)
# sns.barplot(x=coefs.index,y=coefs.values)
# plt.show()

y_pred=log_model.predict(scaled_X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# plot_confusion_matrix(log_model,scaled_X_test,y_test)
# plt.show()
patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]

log_model.predict(patient)
print(log_model.predict_proba(patient))
