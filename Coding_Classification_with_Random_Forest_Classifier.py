import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
# df=pd.read_csv('E:\\Download\\Resource\\penguins_size.csv')
# print(df.isnull().sum())
# print(df.shape)
# df=df.dropna()
# # print(df.isnull().sum())
# print(df.info())

# X=pd.get_dummies(df.drop('species',axis=1),drop_first=True)
# y=df['species']

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
# rfc=RandomForestClassifier(n_estimators=10,max_features=10,random_state=101)

# rfc.fit(X_train,y_train)
# preds=rfc.predict(X_test)
# print(classification_report(y_test,preds))
# print(confusion_matrix(y_test,preds))

df=pd.read_csv('E:\\Download\\Resource\\data-banknote-authentication.csv')
print(df)

# sns.pairplot(df,hue='Class')
# plt.show()

X=df.drop('Class',axis=1)
y=df['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=101)

n_estimators=[64,100,128,200]
max_features=[2,3,4]
bootstrap=[True,False]
oob_score=[True,False]

param_grid={'n_estimators':n_estimators,
            'max_features':max_features,
            'bootstrap':bootstrap,
            'oob_score':oob_score}
# rfc=RandomForestClassifier()
# grid=GridSearchCV(rfc,param_grid)
#
# grid.fit(X_train,y_train)

# print(grid.best_estimator_)
# print(grid.best_params_)

rfc=RandomForestClassifier(n_estimators=200,max_features=2,oob_score=True)
rfc.fit(X_train,y_train)

print(rfc.oob_score_)

predictions=rfc.predict(X_test)

print(classification_report(y_test,predictions))

errors=[]
miscalssfication=[]

for n in range(1,200):
    rfc1=RandomForestClassifier(max_features=2,n_estimators=n)
    rfc1.fit(X_train,y_train)
    pred=rfc1.predict(X_test)
    errs=1-accuracy_score(y_test,pred)
    n_missed=np.sum(pred != y_test)

    errors.append(errs)
    miscalssfication.append(n_missed)


# plt.plot(range(1,200),errors)
# plt.show()

plt.plot(range(1,200),miscalssfication)
plt.show()







