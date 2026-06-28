import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.model_selection import train_test_split
from  sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import  LinearSVC
from sklearn.metrics import classification_report

df=pd.read_csv('E:\\Download\\Resource\\moviereviews.csv')

# print(df)

# **TASK: Check to see if there are any missing values in the dataframe.**

print(df.isnull().sum())

# TASK: Remove any reviews that are NaN

df=df.dropna()
print(df.shape)
#TASK: Check to see if any reviews are blank strings and not just NaN. Note: This means a review text could
# just be: "" or " " or some other larger blank string. How would you check for this? Note: There are many ways!
# Once you've discovered the reviews that are blank strings, go ahead and remove them as well.

df=df[~df['review'].str.isspace()]

print(df.shape)

# TASK: Confirm the value counts per label:

# print(df['label'].value_counts())
count_vect=CountVectorizer(stop_words='english')
matrix = count_vect.fit_transform(df[df['label']=='neg']['review'])
freqs = zip(count_vect.get_feature_names_out(), matrix.sum(axis=0).tolist()[0])
# sort from largest to smallest
print (sorted(freqs, key=lambda x: -x[1])[:20])

#---------- Training and Data--------------------
# TASK: Split the data into features and a label (X and y) and then preform a train/test split.
# You may use whatever settings you like. To compare your results to the solution notebook, use test_size=0.20,
# random_state=101

X=df['review']
y=df['label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=101)

# TASK: Create a PipeLine that will both create a TF-IDF Vector out of the raw text data and fit a
# supervised learning model of your choice. Then fit that pipeline on the training data.
pipe=Pipeline([('tfidf',TfidfVectorizer()),
               ('svc',LinearSVC())])
pipe.fit(X_train,y_train)

preds=pipe.predict(X_test)

print(classification_report(y_test,preds))