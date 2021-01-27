#!/usr/bin/env python
# coding: utf-8

# ##  Problem Statement 

# For a film production company, the success of the film is closely determined by the IMDB ratings the
# movie receives. More often than not, the IMDB rating is influenced by the factors more than the
# story/script like the cast of the film, the director of the film, social media popularity etc. Your task is to
# create a model to predict the IMDB rating of a film before it is released.

# # Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from pandas_profiling import ProfileReport # pip install pandas-profiling[notebook]
import matplotlib.pyplot as plt


# # Importing the dataset

# In[2]:


dataset = pd.read_csv('movie_metadata.csv')
dataset.shape


# # Getting columns with categorical data and numerical data

# In[3]:


#Getting columns with categorical data and numerical data
num_cols = [col for col in dataset.columns if dataset[col].dtype != 'object']
char_cols = [col for col in dataset.columns if dataset[col].dtype == 'object']
num_cols, char_cols


# # Generating correlational matrix 

# In[4]:


corr_matrix = dataset.corr()
plt.figure(figsize= (12,12))
sns.heatmap(corr_matrix, annot=True)


# # Generating an entire report of dataset

# In[5]:


#Generating an entire report of dataset
dataset.profile_report()


# # Data Cleaning

# Removing duplicated values

# In[6]:


#Removing duplicated values
dataset.drop_duplicates(subset='movie_title',keep='first',inplace = True)
dataset.shape


# Checking number of null values in each column

# In[7]:


dataset.isnull().sum()


# Removing all null values in gross since its too much and would affect our prediction in case we fill with other values

# In[8]:


dataset = dataset.dropna(axis = 0, subset = ['gross'])
dataset.shape


# In[9]:


dataset.isnull().sum()


# Removing all null values in budget since its too much and would affect our prediction in case we fill with other values

# In[10]:


dataset = dataset.dropna(axis = 0, subset = ['budget'])
dataset.shape


# In[11]:


dataset.isnull().sum()


# In[12]:


dataset.content_rating.unique()


# Replacing 'Not Rated' in place of null values in content rating

# In[13]:


dataset.content_rating.fillna('Not Rated', inplace = True)


# Filling color,language,content_rating with most frequent value in column

# In[14]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(dataset[['color','language','content_rating']])
dataset[['color','language','content_rating']] = imputer.transform(dataset[['color','language','content_rating']])


# In[15]:


dataset.isnull().sum()


# Filling numerical columns with median value in column

# In[16]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(dataset[num_cols])
dataset[num_cols] = imputer.transform(dataset[num_cols])


# In[17]:


dataset.isnull().sum()


# In[18]:


corr_matrix = dataset.corr()
plt.figure(figsize= (12,12))
sns.heatmap(corr_matrix, annot=True)


# Removing all categorical columns except genre,color,language,content_rating,country since there are many distinct values and is irrelevant for prediction

# In[19]:


dataset = dataset.drop(columns = ['director_name','actor_2_name','actor_1_name','movie_title','actor_3_name','plot_keywords','movie_imdb_link'], axis=1)


# In[20]:


dataset.isnull().sum()


# (Optional) To see entire dataset

# In[21]:


# #To see entire dataset
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
# pd.options.display.max_rows
# pd.set_option('display.max_rows', None)


# # Taking only the top 3 occurances of country, language, content_rating 
It increases the accuracy of the model since the top 3 occurances are too many in these columns
# In[ ]:


top3_country = dataset["country"].value_counts()[:3].index
dataset['country'] = dataset.country.where(dataset.country.isin(top3_country), 'other')


# In[ ]:


top3_language = dataset["language"].value_counts()[:3].index
dataset['language'] = dataset.language.where(dataset.language.isin(top3_language), 'other')


# In[ ]:


top3_content_rating = dataset["content_rating"].value_counts()[:3].index
dataset['content_rating'] = dataset.content_rating.where(dataset.content_rating.isin(top3_content_rating), 'other')


# # Resetting index

# In[22]:


dataset.reset_index(inplace = True, drop = True)


# # Getting unique values in genre

# In[23]:


#Getting unique values in genre
genre = dataset.iloc[:, dataset.columns.get_loc('genres')].values
a = []
unique = [x.split('|') for x in genre]
for x in unique:
    for y in x:
        a.append(y)
a = set(a)


# # Creating null columns for all unique values

# In[24]:


#Creating null columns for all unique values
for x in a:
    dataset[x] = np.nan


# In[25]:


#Encoding the data in the form of 0's and 1's
for x in a:
    for i in range(0,len(dataset)):
        if x in unique[i]:
            dataset[x][i] = 1
        else:
            dataset[x][i] = 0


# # Removing column genre 

# In[26]:


dataset = dataset.drop(columns = ['genres'], axis=1)


# # Encoding color,language,content_rating,country

# In[27]:


dataset = pd.get_dummies(dataset, columns=['color'], prefix = ['color'])


# In[28]:


dataset = pd.get_dummies(dataset, columns=['language'], prefix = ['language'])


# In[29]:


dataset = pd.get_dummies(dataset, columns=['content_rating'], prefix = ['content_rating'])


# In[30]:


dataset = pd.get_dummies(dataset, columns=['country'], prefix = ['country'])


# In[31]:


dataset.head()


# # Defining Matrix of Features

# In[32]:


#Defining Matrix of Features
X = dataset.iloc[:,dataset.columns != 'imdb_score'].values


# # Defining matrix of dependent variable

# In[33]:


#Defining matrix of dependent variable
y = dataset.iloc[:,dataset.columns.get_loc('imdb_score')].values


# # Splitting into training and test set

# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 29)


# # Training the model

# Random Forest Regression(Since it gives maximum accuracy)

# In[35]:


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state=12)
regressor.fit(X_train, y_train)


# # Evaluating the model

# Predicting the output

# In[36]:


y_pred = regressor.predict(X_test)


# Finding accuracy of the model

# In[37]:


score = regressor.score(X_test, y_test)
print(f'Accuracy: {score*100}%')


# In[38]:


rss=((y_test-y_pred)**2).sum()
mse=np.mean((y_test-y_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((y_test-y_pred)**2)))


# Predicted IMDB rating for few random samples in the dataset

# In[39]:


import random
print(f'Actual Value  |  Predicted Value')
print('---------------------------------')
for x in range(0,10):
    num = random.randint(0,100)
    print(f'   {y_test[num]}        |      {y_pred[num]:.1f}')

