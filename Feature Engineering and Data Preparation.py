import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.width',700)
pd.set_option('display.max_columns',11)


df=pd.read_csv('E:\\Download\\Resource\\housing.csv')
# print(df.columns)
# print(df.shape)
# print(df.corr()['SalePrice'].sort_values())
# sns.scatterplot(x='Overall Qual',y='SalePrice',data=df)
# sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df)
# plt.show()

# print(df[(df['Overall Qual'] >8) & (df['SalePrice']<200000)])

# print(df[(df['Gr Liv Area']>4000)&(df['SalePrice']<400000)])

drop_index=df[(df['Gr Liv Area']>4000)&(df['SalePrice']<400000)].index
# print(drop_index)

df=df.drop(drop_index,axis=0)

# sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df)
# plt.show()

df=df.drop(['Unnamed: 0','Order','PID'],axis=1 )
# print(df.isnull().sum())

def precnt_missing(df):
    precent_nan=100 * df.isnull().sum()/len(df)
    precent_nan=precent_nan[precent_nan>0].sort_values()

    return precent_nan

# precent_nan=precnt_missing(df)

# print(precent_nan)

# sns.barplot(x=precent_nan.index,y=precent_nan)
# plt.xticks(rotation=90)
# plt.show()

# print(precent_nan[precent_nan < 1])

# print(df[df['Electrical'].isnull()])
# print(df[df['Bsmt Half Bath'].isnull()])
# print(df[df['Mas Vnr Area'].isnull()])

df=df.dropna(axis=0,subset=['Electrical','Garage Cars'])
# precent_nan=precnt_missing(df)
# print(precent_nan[precent_nan<1])

# sns.barplot(x=precent_nan.index,y=precent_nan)
# plt.xticks(rotation=90)
# plt.ylim(0,1)
# plt.show()

# print(df[df['Bsmt Full Bath'].isnull()])
# print(df[df['Bsmt Half Bath'].isnull()])
# print(df[df['Bsmt Unf SF'].isnull()])


# BSMT Numirec Cols
bsmt_num_cols=['Bsmt Unf SF','Total Bsmt SF','BsmtFin SF 2','BsmtFin SF 1','Bsmt Full Bath','Bsmt Half Bath']

df[bsmt_num_cols]=df[bsmt_num_cols].fillna(0)

# BSMT String COls
bsmt_str_cols=['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols]=df[bsmt_str_cols].fillna('None')

print(df[df['Bsmt Full Bath'].isnull()])


# precent_nan=precnt_missing(df)
# print(precent_nan[precent_nan<1])
print(df.info())
# sns.barplot(x=precent_nan.index,y=precent_nan)
# plt.xticks(rotation=90)
# plt.ylim(0,1)
# plt.show()

df['Mas Vnr Type']=df['Mas Vnr Type'].fillna('None')
df['Mas Vnr Area']=df['Mas Vnr Area'].fillna(0)

# precent_nan=precnt_missing(df)

# sns.barplot(x=precent_nan.index,y=precent_nan)
# plt.xticks(rotation=90)
# plt.ylim(0,1)
# plt.show()


gar_str_cols=['Garage Type','Garage Finish','Garage Qual','Garage Cond']

df[gar_str_cols]=df[gar_str_cols].fillna('None')

df['Garage Yr Blt']=df['Garage Yr Blt'].fillna(0)


df=df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)
# precent_nan=precnt_missing(df)
# #
# # sns.barplot(x=precent_nan.index,y=precent_nan)
# # plt.xticks(rotation=90)
# # plt.show()

df['Fireplace Qu']=df['Fireplace Qu'].fillna('None')

# print(df['Lot Frontage'])

# sns.boxplot(x='Lot Frontage',y='Neighborhood',data=df,orient='h')
# plt.show()

# print(df.groupby('Neighborhood')['Lot Frontage'].mean())

df['Lot Frontage']=df.groupby('Neighborhood')['Lot Frontage'].transform(lambda values:values.mean())

df['Lot Frontage']=df['Lot Frontage'].fillna(0)
# precent_nan=precnt_missing(df)
#
# sns.barplot(x=precent_nan.index,y=precent_nan)
# plt.xticks(rotation=90)
# plt.show()


df['MS SubClass'] = df['MS SubClass'].apply(str)
# print(df.select_dtypes(include='object'))

my_object_df=df.select_dtypes(include='object')
my_numeric_df=df.select_dtypes(exclude='object')




# print(my_object_df)
df_objects_dummies=pd.get_dummies(my_object_df,drop_first=True,dtype=int)

# print(df_objects_dummies)
final_df=pd.concat([my_numeric_df,df_objects_dummies],axis=1)
print(final_df.corr()['SalePrice'].sort_values())



