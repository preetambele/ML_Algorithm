import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score
from mlxtend.plotting import plot_confusion_matrix



df=pd.read_csv('E:\\Download\\Resource\\hearing_test.csv')
print(df.info())
# print(df.value_counts('test_result'))

# sns.countplot(data=df,x='test_result')
# sns.boxplot(x='test_result',y='age',data=df)
# sns.boxplot(x='test_result',y='physical_score',data=df)
# sns.scatterplot(x='age',y='physical_score',data=df,hue='test_result')
# sns.pairplot(data=df,hue='test_result')
# sns.heatmap(df.corr(),annot=True)
# plt.show()

X=df.drop('test_result',axis=1)
y=df['test_result']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=101)

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

log_model=LogisticRegression()
log_model.fit(scaled_X_train,y_train)

print(log_model.coef_)
y_pred=log_model.predict(scaled_X_test)
# print(y_pred)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))