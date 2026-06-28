import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

pd.set_option('display.width',900)
pd.set_option('display.max_column',15)

df=pd.read_csv('E:\\Download\\Resource\\Telco-Customer-Churn.csv')

# print(df.info())
# print(df.describe())
# print(df.isnull().sum())


# ---------TASK:Display the balance of the class labels (Churn) with a Count Plot.-------------------
# sns.countplot(x='Churn',data=df)
# plt.show()

# ---------TASK: Explore the distrbution of TotalCharges between Churn categories with a Box Plot or Violin Plot.----
# sns.boxplot(x='Churn',y='TotalCharges',data=df)
# sns.violinplot(x='Churn',y='TotalCharges',data=df)
# plt.show()

# --------TASK: Create a boxplot showing the distribution of TotalCharges per Contract type, also add in a hue coloring based on the Churn class.--
# plt.figure(figsize=(9,5))
# sns.boxplot(x='Contract',y='TotalCharges',data=df,hue='Churn')
# # plt.legend(bbox_to_anchor=(1.05, 0.7));
# plt.show()

# ------TASK: Create a bar plot showing the correlation of the following features to the class label. Keep in mind,
# for the categorical features, you will need to convert them into dummy variables first, as you can only calculate
# correlation for numeric features
cols_to_consider = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'InternetService','StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod','Churn']
# corr=pd.get_dummies(df[cols_to_consider],drop_first=True).corr()
# print(corr.index)
# # corr_yes_chrun=corr['Churn_Yes'].sort_values().iloc[1:-1]
# print(corr['Churn_Yes'].sort_values())
# # print(corr_yes_chrun)
#
# sns.barplot(x=corr.iloc[:-1].index,y=corr['Churn_Yes'].iloc[:-1])
# plt.xticks(rotation=90);
# plt.show()


# ----------Part 3: Churn Analysis
# This section focuses on segementing customers based on their tenure, creating "cohorts", allowing us
# to examine differences between customer cohort segments.

# ---------------TASK: What are the 3 contract types available?------------

# print(df['Contract'].unique())

# -----TASK: Create a histogram displaying the distribution of 'tenure' column, which is the amount of months a
# customer was or has been on a customer.
# sns.histplot(x='tenure',data=df)
# plt.show()

# -----TASK: Now use the seaborn documentation as a guide to create histograms separated by two additional
# features, Churn and Contract.

# sns.displot(data=df,x='tenure',bins=70,row='Churn',col='Contract')
# plt.show()

# --------------TASK: Display a scatter plot of Total Charges versus Monthly Charges, and color hue by Churn.--------------
# sns.scatterplot(data=df,y='TotalCharges',x='MonthlyCharges',hue='Churn')
# plt.show()

# -----------Creating Cohorts based on Tenure-------------------
# TASK: Treating each unique tenure group as a cohort, calculate the Churn rate (percentage that had Yes Churn) per cohort.
# For example, the cohort that has had a tenure of 1 month should have a Churn rate of 61.99%.
# You should have cohorts 1-72 months with a general trend of the longer the tenure of the cohort,
# the less of a churn rate. This makes sense as you are less likely to stop service the longer you've had it.

# yes_churn=df.groupby(['Churn','tenure']).count().transpose()['Yes']
# no_churn=df.groupby(['Churn','tenure']).count().transpose()['No']
# churn_rate=100 * yes_churn/(no_churn + yes_churn)

# churn_rate.transpose()['customerID'].plot()
# plt.show()

# ----------Broader Cohort Groups----------------
# TASK: Based on the tenure column values, create a new column called Tenure Cohort that creates 4 separate categories:
#
# '0-12 Months'
# '12-24 Months'
# '24-48 Months'
# 'Over 48 Months'

def cohort(tenure):
    if tenure<12:
        return '0-12 Months'
    elif tenure<25:
        return '12-24 Months'
    elif tenure<49:
        return '24-48 Months'
    else:
        return 'Over 48 Months'

# df['Tenure_Cohort']=df['tenure'].apply(cohort)

# TASK: Create a scatterplot of Total Charges versus Monthly Charts,colored by Tenure Cohort defined in the previous task.
# sns.scatterplot(y='TotalCharges',x='MonthlyCharges',data=df,hue='Tenure_Cohort')
# plt.show()

# TASK: Create a count plot showing the churn count per cohort.**
# sns.countplot(x='Tenure_Cohort',data=df,hue='Churn')
# plt.show()

# TASK: Create a grid of Count Plots showing counts per Tenure Cohort, separated out by contract type and
# colored by the Churn hue.

# sns.catplot(data=df,x='Tenure_Cohort',hue='Churn',kind='count',col='Contract')
# plt.show()

# ------Part 4: Predictive Modeling----------------

# TASK : Separate out the data into X features and Y label. Create dummy variables where necessary and note
# which features are not useful and should be dropped

X = df.drop(['Churn','customerID'],axis=1)
X = pd.get_dummies(X,drop_first=True)
# print(X)
y = df['Churn']
print(type(df['Churn']))
# print(pd.get_dummies(df['Churn'],drop_first=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
# TASK: Decision Tree Perfomance. Complete the following tasks:
    # Train a single decision tree model (feel free to grid search for optimal hyperparameters).
    # Evaluate performance metrics from decision tree, including classification report and plotting a confusion matrix.
    # Calculate feature importances from the decision tree.
    # OPTIONAL: Plot your tree, note, the tree could be huge depending on your pruning, so it may crash your
    # notebook if you display it with plot_tree.

dt_model=DecisionTreeRegressor(max_depth=6)

# param_grid= {'criterion':['gini','entropy'],
#               'max_depth':list(range(1,11))}
#
# dt_grid=GridSearchCV(dt_model,param_grid)
# dt_grid.fit(X_train,y_train)

# print(dt_grid.best_params_)
# dt_model.fit(X_train,y_train)
# preds=dt_model.predict(X_test)
#
# print(classification_report(y_test,preds))











