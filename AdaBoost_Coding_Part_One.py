import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('E:\\Download\\Resource\\mushrooms.csv')
print(df.info())

# sns.countplot(x='class',data=df)
# plt.show()

# feat_uni=df.describe().transpose().reset_index().sort_values('unique')
# plt.figure(figsize=(6,5))
# sns.barplot(x='index',y='unique',data=feat_uni)
# plt.xticks(rotation=90)
# plt.show()

X=df.drop('class',axis=1)
X=pd.get_dummies(X,drop_first=True)

y=df['class']

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.15,random_state=101)
# model=AdaBoostClassifier(n_estimators=1)
# model.fit(X_train,y_train)
# predications=model.predict(X_test)

# print(classification_report(y_test,predications))


# errors_rates=[]
# for n in range(1,96):
#     model=AdaBoostClassifier(n_estimators=n)
#     model.fit(X_train,y_train)
#     predict=model.predict(X_test)
#     err=1 - accuracy_score(y_test,predict)
#     errors_rates.append(err)
#
# plt.plot(range(1,96),errors_rates)
# plt.show()

#----------------------------Gradient Boosting Coding--------------------------------------------
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.15,random_state=101)

param_grid={'n_estimators':[50,100],
            'learning_rate':[0.1,0.05,0.2],
            'max_depth':[3,4,5]}

gb_model=GradientBoostingClassifier()

grid=GridSearchCV(gb_model,param_grid)
grid.fit(X_train,y_train)

predications=grid.predict(X_test)

print(grid.best_params_)

print(classification_report(y_test,predications))