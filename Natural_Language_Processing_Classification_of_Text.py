import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.width',900)
pd.set_option('display.max_column',15)

df=pd.read_csv('E:\\Download\\Resource\\airline-tweets.csv')

print(df)

# sns.countplot(data=df,x='airline_sentiment')
# sns.countplot(data=df,x='negativereason')
# plt.xticks(rotation=90)

# sns.countplot(data=df,x='airline',hue='airline_sentiment')
# plt.show()

X=df['text']
y=df['airline_sentiment']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=101)

tfidf=TfidfVectorizer(stop_words='english')
tfidf.fit(X_train)
X_train_tfidf=tfidf.transform(X_train)

X_test_tfidf=tfidf.transform(X_test)

# ----naive_bayes import MultinomialNB----
nb=MultinomialNB()
nb.fit(X_train_tfidf,y_train)

#----LogisticRegression----
log_model=LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf,y_train)

# -----SVC-----------
rbf_svc=SVC()
rbf_svc.fit(X_train_tfidf,y_train)

# ----Linear SVC
linear_svc=LinearSVC()
linear_svc.fit(X_train_tfidf,y_train)

def report(model):
    preds=model.predict(X_test_tfidf)
    print(classification_report(y_test,preds))
    print(confusion_matrix(y_test,preds))
    # matrix=confusion_matrix(model,y_test)
    # disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    # disp.plot()
    # plt.show()

# report(nb)
# report(log_model)






