import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from svm_margin_plot import plot_svm_boundary
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('E:\\Download\\Resource\\09_Support_Vector_Machines\\mouse_viral_study.csv')
print(df)

# sns.scatterplot(x='Med_1_mL',y='Med_2_mL',data=df,hue='Virus Present')
# x=np.linspace(0,10,100)
# m=-1
# b=11
# y=m*x + b
# plt.plot(x,y,'black')
# plt.show()
y=df['Virus Present']
X=df.drop('Virus Present',axis=1)
# model=SVC(kernel='linear',C=10000)
# model.fit(X,y)
# plot_svm_boundary(model,X,y)

# model1=SVC(kernel='linear',C=0.05)
# model1.fit(X,y)
# plot_svm_boundary(model1,X,y)

# model=SVC(kernel='rbf',C=1,gamma='scale')
# model.fit(X,y)
# plot_svm_boundary(model, X, y)

# model=SVC(kernel='sigmoid')
# model.fit(X,y)
# plot_svm_boundary(model, X, y)

model=SVC(kernel='poly',degree=1,C=1)
model.fit(X, y)
plot_svm_boundary(model, X, y)

svm=SVC()
param_grid={'C':[0.01,0.1,1],'krenel':['linear','rbf']}

grid=GridSearchCV(svm,param_grid)

grid.fit(X,y)

# plot_svm_boundary(grid,X, y)

