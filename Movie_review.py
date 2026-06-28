import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width',700)
pd.set_option('display.max_column',9)


fandnag = pd.read_csv('E:\\Download\\Resource\\05_Capstone_Project\\fandango_scrape.csv')

# print(fandnag)
# print(fandnag.info())
# print(fandnag.describe())
# print(fandnag.isnull().sum())

# sns.scatterplot(data=fandnag,x='RATING',y='VOTES')
# plt.show()


# print(fandnag.corr())

# TASK: Assuming that every row in the FILM title column has the same format:
#       Film Title Name (Year)

# title=' Film Title Name (Year)'
# print(title.split('(')[-1].replace(')',''))
# Create a new column that is able to strip the year from the title strings and set this new column as YEAR
fandnag['Year']=fandnag['FILM'].apply(lambda title:title.split('(')[-1].replace(')',''))
# print(fandnag)


# TASK: How many movies are in the Fandango DataFrame per year?
# print(fandnag['Year'].value_counts())

# TASK: Visualize the count of movies per year with a plot:
# sns.countplot(data=fandnag,x='Year')
# plt.show()

# TASK: What are the 10 movies with the highest number of votes?
# print(fandnag.sort_values('VOTES',ascending=False))
# print(fandnag.nlargest(10,'VOTES'))

# TASK: How many movies have zero votes?
# print(len(fandnag[fandnag['VOTES']==0]))
# print(fandnag[fandnag['VOTES']==0]['FILM'].count())

# TASK: Create DataFrame of only reviewed films by removing any films that have zero votes.

fan_reviewed=fandnag[fandnag['VOTES'] >0]
# fan_reviewed=fandnag[~(fandnag['VOTES']==0)]
# print(fandnag.shape)
# print(fan_reviewed.shape)

# TASK: Create a KDE plot (or multiple kdeplots) that displays the distribution of ratings that are displayed (STARS) versus what the true rating was from votes (RATING). Clip the KDEs to 0-5.
# plt.figure(figsize=(10,4))
# sns.kdeplot(data=fan_reviewed,x='RATING',clip=[0,5],fill=True,label='True_Rating')
# sns.kdeplot(data=fan_reviewed,x='STARS',clip=[0,5],fill=True,label='Stars Display')
# plt.legend(loc=(1.05,0.5))
# plt.show()


# TASK: Let's now actually quantify this discrepancy. Create a new column of the different between STARS displayed versus true RATING. Calculate this difference with STARS-RATING and round these differences to the nearest decimal point.
fan_reviewed['Star_diff']=fan_reviewed['STARS']-fan_reviewed['RATING']
fan_reviewed['Star_diff']=fan_reviewed['Star_diff'].round(2)
# print(fan_reviewed['Star_diff'])

# TASK: Create a count plot to display the number of times a certain difference occurs:
# plt.figure(figsize=(12,4))
# sns.countplot(x='Star_diff',data=fan_reviewed,palette='magma')
# plt.show()

# TASK: We can see from the plot that one movie was displaying over a 1 star difference than its true rating! What movie had this close to 1 star differential?
# print(fan_reviewed[fan_reviewed['Star_diff']==1])


# Part Three: Comparison of Fandango Ratings to Other Sites

# TASK: Read in the "all_sites_scores.csv" file by running the cell below
all_sites=pd.read_csv('E:\\Download\\Resource\\05_Capstone_Project\\all_sites_scores.csv')
# print(all_sites.info())

# TASK: Create a scatterplot exploring the relationship between RT Critic reviews and RT User reviews.
# sns.scatterplot(data=all_sites,x='RottenTomatoes_User',y='RottenTomatoes')
# plt.ylim(0,100)
# plt.xlim(0,100)
# plt.show()


# TASK: Create a new column based off the difference between critics ratings and users ratings for Rotten Tomatoes. Calculate this with RottenTomatoes-RottenTomatoes_User

all_sites['Rotten_Diff']=all_sites['RottenTomatoes']-all_sites['RottenTomatoes_User']
# print(all_sites['Rotten_Diff'])

# TASK: Calculate the Mean Absolute Difference between RT scores and RT User scores as described above.
# print(all_sites['Rotten_Diff'].apply(abs).mean())

# TASK: Plot the distribution of the differences between RT Critics Score and RT User Score. There should be negative values in this distribution plot. Feel free to use KDE or Histograms to display this distribution.
# sns.kdeplot(data=all_sites,x='RottenTomatoes',y='RottenTomatoes_User')
# sns.histplot(data=all_sites,x='Rotten_Diff',kde=True,bins=25)
# plt.show()

# TASK: Now create a distribution showing the absolute value difference between Critics and Users on Rotten Tomatoes.
# sns.histplot(x=all_sites['Rotten_Diff'].apply(abs),kde=True,bins=25)
# plt.show()

# TASK: What are the top 5 movies users rated higher than critics on average:
# print(all_sites.nsmallest(5,'Rotten_Diff'))

# TASK: Now show the top 5 movies critics scores higher than users on average.
# print(all_sites.nlargest(5,'Rotten_Diff'))

# TASK: Display a scatterplot of the Metacritic Rating versus the Metacritic User rating.
# sns.scatterplot(data=all_sites,x='Metacritic',y='Metacritic_User')
# plt.xlim(0,100)
# plt.ylim(0,10)
# plt.show()

# TASK: Create a scatterplot for the relationship between vote counts on MetaCritic versus vote counts on IMDB.
# sns.scatterplot(data=all_sites,x='Metacritic_user_vote_count',y='IMDB_user_vote_count')
# plt.show()

# Notice there are two outliers here. The movie with the highest vote count on IMDB only has about 500 Metacritic ratings. What is this movie?
# TASK: What movie has the highest IMDB user vote count?
# print(all_sites.sort_values('IMDB_user_vote_count',ascending=False).head(1))
# print(all_sites.nlargest(1,'IMDB_user_vote_count'))

# TASK: What movie has the highest Metacritic User Vote count?
# print(all_sites.nlargest(1,'Metacritic_user_vote_count'))
# print(all_sites.sort_values('Metacritic_user_vote_count',ascending=False).head(1))

# TASK: Combine the Fandango Table with the All Sites table. Not every movie in the Fandango table is in the All Sites
# table, since some Fandango movies have very little or no reviews. We only want to compare movies that are in
# both DataFrames, so do an inner merge to merge together both DataFrames based on the FILM columns.

rating=pd.merge(fandnag,all_sites,on='FILM',how='inner')
# print(rating.info())

# TASK: Create new normalized columns for all ratings so they match up within the 0-5 star range shown on Fandango.
# There are many ways to do this.
# print(rating.describe().transpose()['max'])

rating['RT_Norm']=np.round(rating['RottenTomatoes']/20,1)
rating['RTU_Norm']=np.round(rating['RottenTomatoes_User']/20,1)

rating['Meta_Norm']=np.round(rating['Metacritic']/20,1)
rating['Meta_U_Norm']=np.round(rating['Metacritic_User']/2,1)

rating['IMDB_Norm']=np.round(rating['IMDB']/2,1)
# print(rating.columns)

# TASK: Now create a norm_scores DataFrame that only contains the normalizes ratings.
# Include both STARS and RATING from the original Fandango table.
norm_scores=rating[['FILM','STARS','RATING','RT_Norm', 'RTU_Norm', 'Meta_Norm', 'Meta_U_Norm', 'IMDB_Norm']]
# norm_scores.drop(norm_scores[(norm_scores['FILM']=='Fifty Shades of Grey (2015)')].index,inplace=True)
# print(norm_scores.drop(norm_scores.iloc[0:1].index))
print(norm_scores)

# TASK: Create a plot comparing the distributions of normalized ratings across all sites. There are many ways to do this,
# but explore the Seaborn KDEplot docs for some simple ways to quickly show this. Don't worry if your plot format does
# not look exactly the same as ours, as long as the differences in distribution are clear.
# plt.figure(figsize=(10,4))
# sns.kdeplot(data=norm_scores,clip=[0,5],shade=True,palette='Set1')
# plt.show()


# Clearly Fandango has an uneven distribution. We can also see that RT critics have the most uniform distribution.
# Let's directly compare these two.

# TASK: Create a KDE plot that compare the distribution of RT critic ratings against the STARS displayed by Fandango.

# sns.kdeplot(data=norm_scores[['RT_Norm','STARS']],clip=[0,5],shade=True,palette='Set1')
# plt.show()

# OPTIONAL TASK: Create a histplot comparing all normalized scores.
# sns.histplot(data=norm_scores,bins=25)
# plt.show()

# How are the worst movies rated across all platforms?
# TASK: Create a clustermap visualization of all normalized scores. Note the differences in ratings, highly rated movies
# should be clustered together versus poorly rated movies. Note: This clustermap does not need to have the FILM titles
# as the index, feel free to drop it for the clustermap.

# norm_scores.set_index('FILM',inplace=True)
# print(norm_scores.set_index('FILM'))
# sns.clustermap(data=norm_scores,cmap='magma',col_cluster=False)
# plt.show()

# TASK: Clearly Fandango is rating movies much higher than other sites, especially considering that it is then displaying
# a rounded up version of the rating. Let's examine the top 10 worst movies. Based off the Rotten Tomatoes Critic Ratings,
# what are the top 10 lowest rated movies? What are the normalized scores across all platforms for these movies?
# You may need to add the FILM column back in to your DataFrame of normalized scores to see the results.

worst_films=norm_scores.nsmallest(10,'RT_Norm')

# FINAL TASK: Visualize the distribution of ratings across all sites for the top 10 worst movies.
# 
# sns.kdeplot(data=worst_films,clip=[0,5],shade=True,palette='Set1')
# plt.show()



