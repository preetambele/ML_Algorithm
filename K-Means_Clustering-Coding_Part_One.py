import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df=pd.read_csv('E:\\Download\\Resource\\bank-full.csv')

print(df.info())

# sns.histplot(data=df,x='age',bins=30,kde=True,hue='loan')
# sns.histplot(data=df,x='pdays',bins=30)
# sns.histplot(data=df,x='duration',hue='contact')
# plt.xlim(0,1000)
# print(df['job'].value_counts().index)
# plt.figure(figsize=(7,4))
# sns.countplot(data=df,x='job',order=df['job'].value_counts().index,hue='default')
# plt.xticks(rotation=90)
# plt.show()

X=pd.get_dummies(df)
# print(X)
scaler=StandardScaler()
scaled_X=scaler.fit_transform(X)


# model=KMeans(n_clusters=2)
# cluster_labels=model.fit_predict(scaled_X)
# print(cluster_labels)

# X['Cluster']=cluster_labels

# X_Corr=X.corr()['Cluster'].iloc[:-1].sort_values()
# print(X_Corr)
# X_Corr.plot(kind='bar')
# plt.show()

ssd=[]
for k in range(2,10):
    kmodel=KMeans(n_clusters=k)
    kmodel.fit_transform(scaled_X)

    ssd.append(kmodel.inertia_)

# print(ssd)

# plt.plot(range(2,10),ssd,'o--')
# plt.show()

ser=pd.Series(ssd)
print(ser.diff())