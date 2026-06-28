import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


text = ['This is a line',
           "This is another line",
       "Completely different line"]


# cv=CountVectorizer()
# sparse_matrix=cv.fit(text)

# print(cv.vocabulary_)

tfidf=TfidfTransformer()
cv = CountVectorizer()
counts=cv.fit_transform(text)
result=tfidf.fit_transform(counts)

print(result.todense())

tv=TfidfVectorizer()

tv_result=tv.fit_transform(text)
print(tv_result.todense())