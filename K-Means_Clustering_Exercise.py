import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.width',900)
pd.set_option('display.max_column',16)

df=pd.read_csv('E:\\Download\\Resource\\CIA-Country-Facts.csv')

print(df.info())
# Exploratory Data Analysis
# TASK: Explore the rows and columns of the data as well as the data types of the columns.
# print(df.info())
# print(df.describe().transpose())

# TASK: Create a histogram of the Population column.
# sns.histplot(data=df,x='Population')
# plt.show()

# TASK: You should notice the histogram is skewed due to a few large countries, reset the X axis to only
# show countries with less than 0.5 billion people

# print(df[df['Population']<=500000000])
# sns.histplot(data=df[df['Population']<=500000000],x='Population')
# plt.show()

# TASK: Now let's explore GDP and Regions. Create a bar chart showing the mean GDP per Capita per region
# (recall the black bar represents std).

# print(df['Region'])
# plt.figure(figsize=(7,5))
# sns.barplot(data=df,x='Region',y='GDP ($ per capita)',estimator=np.mean)
# plt.xticks(rotation=90)
# plt.show()

# TASK: Create a scatterplot showing the relationship between Phones per 1000 people and the GDP per Capita.
# Color these points by Region.

# print(df['Phones (per 1000)'])
# plt.figure(figsize=(10,5))
# sns.scatterplot(data=df,x='Phones (per 1000)',y='GDP ($ per capita)',hue='Region')
# plt.legend(loc=(1.05,0.5))
# plt.show()


# TASK: Create a scatterplot showing the relationship between GDP per Capita and Literacy (color the points by Region).
# What conclusions do you draw from this plot?

# sns.scatterplot(data=df,x='Literacy (%)',y='GDP ($ per capita)',hue='Region')
# plt.legend(loc=(0.7,0.7))
# plt.show()

# TASK: Create a Heatmap of the Correlation between columns in the DataFrame.

df_corr=df.drop(['Country','Region'],axis=1)
# sns.heatmap(df_corr.corr())
# plt.show()

# TASK: Seaborn can auto perform hierarchal clustering through the clustermap() function.
# Create a clustermap of the correlations between each column with this function.

# sns.clustermap(df_corr.corr())
# plt.show()

# ---------Data Preparation and Model Discovery-----------------------

# Missing Data
# TASK: Report the number of missing elements per column.

# print(df.isnull().sum().sort_values(ascending=False))

# TASK: What countries have NaN for Agriculture? What is the main aspect of these countries?

# print(df[df['Agriculture'].isnull()])

# TASK: You should have noticed most of these countries are tiny islands, with the exception of Greenland and
# Western Sahara. Go ahead and fill any of these countries missing NaN values with 0, since they are so small or
# essentially non-existant. There should be 15 countries in total you do this for.
# For a hint on how to do this, recall you can do the following:

df[df['Agriculture'].isnull()]=df[df['Agriculture'].isnull()].fillna(0)

# print(df[df['Agriculture'].isnull()]['Country'])
print(df.isnull().sum().sort_values(ascending=False))

# TASK: Notice climate is missing for a few countries, but not the Region! Let's use this to our advantage.
# Fill in the missing Climate values based on the mean climate value for its region.

print(df[df['Climate'].isnull()])
# print(df[df['Country']=='Angola'])