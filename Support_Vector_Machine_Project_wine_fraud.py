import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv('E:\\Download\\Resource\\09_Support_Vector_Machines\\wine_fraud.csv')
# print(df.info())

# -----TASK: What are the unique variables in the target column we are trying to predict (quality)?----------

print(df['quality'].unique())

# TASK: Create a countplot that displays the count per category of Legit vs Fraud. Is the label/target balanced or unbalanced?---------

# sns.countplot(x='quality',data=df,hue='quality')
# plt.show()

# TASK: Let's find out if there is a difference between red and white wine when it comes to fraud. Create a countplot that has the wine type on the x axis with the hue separating columns by Fraud vs Legit.
print(df['type'].unique())
# sns.countplot(x='type',data=df,hue='quality')
# plt.show()

# TASK: What percentage of red wines are Fraud? What percentage of white wines are fraud?
reds=df[df['type'] =='red']
whites=df[df['type'] =='white']

# print(len(reds[reds['quality'] =='Fraud'])/len(reds) * 100)
# print(len(whites[whites['quality'] =='Fraud'])/len(whites)* 100)

# TASK: Calculate the correlation between the various features and the "quality" column.
# To do this you may need to map the column to 0 and 1 instead of a string.

# df['Fraud']= df['quality'].map({'Legit':0,'Fraud':1})

# df1=df.drop(['quality','type'],axis=1)
# print(df1.corr()['Fraud'])

# TASK: Create a bar plot of the correlation values to Fraudlent wine.

# sns.barplot(df1.corr()['Fraud'][:-1].sort_values())
# plt.xticks(rotation=90)
# plt.show()

# TASK: Create a clustermap with seaborn to explore the relationships between variables.
# sns.clustermap(df1.corr())
# plt.show()

# Machine Learning Model
# TASK: Convert the categorical column "type" from a string or "red" or "white" to dummy variables:
print(df.columns)
df['type']=pd.get_dummies(df['type'],drop_first=True)

# TASK: Separate out the data into X features and y target label ("quality" column)
X=df.drop('quality',axis=1)
y=df['quality']

# TASK: Perform a Train|Test split on the data, with a 10% test size. Note: The solution uses a random state of 101
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=101)

# TASK: Scale the X train and X test data.

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

#TASK: Create an instance of a Support Vector Machine classifier. Previously we have left this model "blank",
# (e.g. with no parameters). However, we already know that the classes are unbalanced, in an attempt to help alleviate
# this issue, we can automatically adjust weights inversely proportional to class frequencies in the input data
# with a argument call in the SVC() call. Check out the [documentation for SVC] online and look up what the
# argument\parameter is.

svc=SVC(class_weight='balanced')

# TASK: Use a GridSearchCV to run a grid search for the best C and gamma parameters.
param_grid= {'C':[0.001,0.01,0.1,0.5,1],
             'gamma':['scale','auto']}

grid=GridSearchCV(svc,param_grid)

grid.fit(scaled_X_train,y_train)

# print(grid.best_params_)

# TASK: Display the confusion matrix and classification report for your model.
grid_pred=grid.predict(scaled_X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))