import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from  sklearn.model_selection import GridSearchCV


df=pd.read_csv('E:\\Download\\Resource\\gene_expression.csv')

print(df)

# sns.scatterplot(data=df,x='Gene One',y='Gene Two',hue='Cancer Present')
# sns.pairplot(data=df,hue='Cancer Present')
# plt.show()

X=df.drop('Cancer Present',axis=1)
y=df['Cancer Present']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

# knn_model=KNeighborsClassifier(n_neighbors=1)
# knn_model.fit(scaled_X_train,y_train)

# y_pred=knn_model.predict(scaled_X_test)
# print(len(y_test))
# # print(confusion_matrix(y_test,y_pred))
# # print(classification_report(y_test,y_pred))
# # print(accuracy_score(y_test,y_pred))

# test_error_rate=[]
#
# for k in range(1,30):
#     knn_model=KNeighborsClassifier(n_neighbors=k)
#     knn_model.fit(scaled_X_train,y_train)
#
#     y_pred=knn_model.predict(scaled_X_test)
#     test_error=1 - accuracy_score(y_test,y_pred)
#     test_error_rate.append(test_error)
knn=KNeighborsClassifier()
# print(test_error_rate)
# plt.plot(range(1,30),test_error_rate)
# plt.ylabel('Error Rate')
# plt.xlabel('K Neighbors')
# plt.show()

operation=[('scaler',scaler),('knn',knn)]

pipe=Pipeline(operation)
k_values=list(range(1,20))

param_grid={'knn__n_neighbors':k_values}

full_cv_classifier=GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')

full_cv_classifier.fit(X_train,y_train)
# print(full_cv_classifier.best_estimator_.get_params())
full_prediction=full_cv_classifier.predict(X_test)

print(classification_report(y_test,full_prediction))



