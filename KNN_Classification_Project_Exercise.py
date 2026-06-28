import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix


df=pd.read_csv('E:\\Download\\Resource\\08_K_Nearest_Neighbors\\sonar.all-data.csv')

# print(df.info())

# ----------TASK: Create a heatmap of the correlation between the difference frequency responses----

# sns.heatmap(df.corr(),cmap='coolwarm')
# plt.show()

# -------------------TASK: What are the top 5 correlated frequencies with the target\label?-------------------
df['Target']=df['Label'].map({'R':0,'M':1})
# print(df)
# df=df.drop('Label',axis=1)
# print(df.corr()['Target'].sort_values())

# TASK: Split the data into features and labels, and then split into a training set and test set, with 90% for Cross-Validation training, and 10% for a final test set.

X=df.drop(['Target','Label'],axis=1)
y=df['Label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

# TASK: Create a PipeLine that contains both a StandardScaler and a KNN model
scaler=StandardScaler()
knn=KNeighborsClassifier()

operation=[('scaler',scaler),('knn',knn)]

pipe=Pipeline(operation)

# TASK: Perform a grid-search with the pipeline to test various values of k and report back the best performing parameters.
k_values=list(range(1,30))

param_grid={'knn__n_neighbors':k_values}

full_cv_classifier=GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')
full_cv_classifier.fit(X_train,y_train)
# print(full_cv_classifier.best_estimator_.get_params())

# pd.DataFrame(full_cv_classifier.cv_results_)['mean_test_score'].plot()
# plt.show()

y_pred=full_cv_classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))