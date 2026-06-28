import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.width',900)
pd.set_option('display.max_column',15)

df=pd.read_csv('E:\\Download\\Resource\\Telco-Customer-Churn.csv')

X=df.drop(['Churn','customerID'],axis=1)
X=pd.get_dummies(X,drop_first=True)
# print(X)
y=df['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=101)

# TASK: Decision Tree Perfomance. Complete the following tasks:
    # Train a single decision tree model (feel free to grid search for optimal hyperparameters).
    # Evaluate performance metrics from decision tree, including classification report and plotting a confusion matrix.
    # Calculate feature importances from the decision tree.
    # OPTIONAL: Plot your tree, note, the tree could be huge depending on your pruning, so it may crash your
    # notebook if you display it with plot_tree.

dt_model=DecisionTreeClassifier()
# dt_model.fit(X_train,y_train)
#
# preds=dt_model.predict(X_test)
# print(classification_report(y_test,preds))

# param_grid= {'criterion':['gini','entropy'],
#               'max_depth':list(range(1,11))}
#
# dt_grid=GridSearchCV(dt_model,param_grid)
# dt_grid.fit(X_train,y_train)
# print(grid_model.best_params_)
# grid_pred=dt_grid.predict(X_test)
# print(classification_report(y_test,grid_pred))

# matrix=confusion_matrix(y_test, grid_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
# # Then just plot it:
# disp.plot()
# # And show it:
# plt.show()

# feat_imp=pd.DataFrame(data=dt_grid.best_estimator_.feature_importances_,index=X.columns,columns=['Feat Imp'])
# feat_imp= feat_imp.sort_values('Feat Imp')
# feat_imp=feat_imp[feat_imp['Feat Imp']>0]
# print(feat_imp)

# sns.barplot(data=feat_imp,x=feat_imp.index,y='Feat Imp')
# plt.xticks(rotation=90)
# plt.show()

# plot_tree(dt_grid.best_estimator_,filled=True,feature_names=X.columns)
# plt.show()

# ------------------Random Forest-----------------------------------------
# TASK: Create a Random Forest model and create a classification report and confusion matrix from its predicted
# results on the test set.

# rf_model=RandomForestClassifier()

# n_estimators=[64,100,128,200]
# param_grid={'n_estimators':n_estimators}
# rf_grid=GridSearchCV(rf_model,param_grid)
#
# rf_grid.fit(X_train,y_train)
#
# rf_pred=rf_grid.predict(X_test)
# print(classification_report(y_test,rf_pred))

# matrix=confusion_matrix(y_test,rf_pred)
# disp=ConfusionMatrixDisplay(confusion_matrix=matrix)
# disp.plot()
# plt.show()


#- Boosted Trees
# TASK: Use AdaBoost or Gradient Boosting to create a model and report back the classification report and
# plot a confusion matrix for its predicted results

ab=AdaBoostClassifier()
param_grid = {'n_estimators':list(range(1,34))}
ab_grid=GridSearchCV(ab,param_grid)
ab_grid.fit(X_train,y_train)
ab_pred=ab_grid.predict(X_test)
print(classification_report(y_test,ab_pred))

matrix=confusion_matrix(y_test,ab_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot()
plt.show()

