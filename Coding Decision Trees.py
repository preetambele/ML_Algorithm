import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
pd.set_option('display.width',700)
pd.set_option('display.max_column',11)


df=pd.read_csv('E:\\Download\\Resource\\penguins_size.csv')

# print(df.info())
# print(df.isnull().sum())

df=df.dropna()
print(df.info())

# print(df['sex'].unique())
#
# print(df[df['sex'] =='.'])
# print(df[df['species']=='Gentoo'].groupby('sex').describe().transpose())

df.at[336,'sex']='FEMALE'

# print(df.loc[336])

# sns.pairplot(df,hue='species')
# sns.catplot(x='species',y='culmen_length_mm',data=df,kind='box',col='sex')
# plt.show()

X=pd.get_dummies(df.drop('species',axis=1),drop_first=True)
y=df['species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)

model=DecisionTreeClassifier()

model.fit(X_train,y_train)
# base_pred=model.predict(X_test)
#
# print(classification_report(y_test,base_pred))
# plt.figure(figsize=(10,5),dpi=200)
# plot_tree(model,feature_names=X.columns,filled=True)
# plt.show()


def report_model(model):
    model_pred=model.predict(X_test)
    print(classification_report(y_test,model_pred))
    print('\n')
    plt.figure(figsize=(5, 4), dpi=200)
    plot_tree(model, feature_names=X.columns, filled=True)
    plt.show()

# report_model(model)
# pruned_tree=DecisionTreeClassifier(max_depth=2)
# pruned_tree.fit(X_train,y_train)
# report_model(pruned_tree)

# max_leaf_tree=DecisionTreeClassifier(max_leaf_nodes=3)
# max_leaf_tree.fit(X_train,y_train)
# report_model(max_leaf_tree)

entropy_tree=DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train,y_train)
report_model(entropy_tree)



