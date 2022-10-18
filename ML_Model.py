#!/usr/bin/env python
# coding: utf-8

# # Part A: Hit or No Hit?
# __Logistic regression__  models to determine if a song would be a hit in a given decade based on a bank of 5000+ songs

# #### Run a preliminary logistic regression model to understand the coefficients of song attributes in merged dataset of all decades

# In[1]:


# Logistic Regression
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


data60s = pd.read_csv('dataset-of-60s.csv')
data70s = pd.read_csv('dataset-of-70s.csv')
data80s = pd.read_csv('dataset-of-80s.csv')
data90s = pd.read_csv('dataset-of-90s.csv')
data00s = pd.read_csv('dataset-of-00s.csv')
data10s = pd.read_csv('dataset-of-10s.csv')


# In[3]:


data60s['decade']=1960
data70s['decade']=1970
data80s['decade']=1980
data90s['decade']=1990
data00s['decade']=2000
data10s['decade']=2010


# In[4]:


merged_data = pd.concat([data60s,data70s,data80s,data90s,data00s,data10s],axis=0)
merged_data['decade']=pd.to_datetime(merged_data['decade'], format='%Y')
merged_data


# In[5]:


# Summary Statistics
merged_data.describe()


# In[6]:


# shows that data is evenly split
merged_data['target'].value_counts()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(30,18))

loc1 = fig.add_subplot(2, 2, 1)

sns.distplot(merged_data['danceability'],ax=loc1,label='Danceability')
sns.distplot(merged_data['valence'],ax=loc1,label='Valence')
sns.distplot(merged_data['liveness'],ax=loc1,label='Liveness')
sns.distplot(merged_data['acousticness'],ax=loc1,label='Acousticness')


loc1.set_title("Music Feature Score Distributions\n",fontsize=16)
#loc2.set_title("Valence Score Distribution")
#loc3.set_title("Liveness Score Distribution")
#loc4.set_title("Acousticness Score Distribution")

plt.xlabel("Feature Scores")

plt.legend(loc='upper right',prop={'size':10})


# In[8]:


b=pd.DataFrame(merged_data.groupby(['decade','target'])['tempo'].mean())
b.reset_index(inplace=True)
sns.lineplot(data=b,x='decade',y='tempo',hue='target')

plt.title('Change in BPM Across Decades')
plt.legend(['flop','hit'])


# In[9]:


#Logistic Regression model on merged decade data to understand audio feature coefficients
import statsmodels.formula.api as smf

LR_results = smf.logit('target ~ danceability + energy + key + loudness + mode + speechiness + acousticness + instrumentalness + liveness + valence + tempo + time_signature + chorus_hit', data=merged_data).fit()
                       
LR_results.summary()


# In[10]:


# odds ratio to calculate most influential factors
odds_ratio = np.exp(LR_results.params)
odds_ratio


# #### Logistic Regression Model Building

# In[11]:


# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[12]:


# Upload all decade song data files
data60s = pd.read_csv('dataset-of-60s.csv')
data70s = pd.read_csv('dataset-of-70s.csv')
data80s = pd.read_csv('dataset-of-80s.csv')
data90s = pd.read_csv('dataset-of-90s.csv')
data00s = pd.read_csv('dataset-of-00s.csv')
data10s = pd.read_csv('dataset-of-10s.csv')


# In[13]:


# Drop non-numeric data 
data60s.drop(['track', 'artist', 'uri','chorus_hit','sections'], axis=1, inplace=True) #13 cols
data70s.drop(['track', 'artist', 'uri','chorus_hit','sections'], axis=1, inplace=True) #13 cols
data80s.drop(['track', 'artist', 'uri','chorus_hit','sections'], axis=1, inplace=True) #13 cols
data90s.drop(['track', 'artist', 'uri','chorus_hit','sections'], axis=1, inplace=True) #13 cols
data00s.drop(['track', 'artist', 'uri','chorus_hit','sections'], axis=1, inplace=True) #13 cols
data10s.drop(['track', 'artist', 'uri','chorus_hit','sections'], axis=1, inplace=True) #13 cols


unscaled_data60 = data60s.iloc[:,0:-1]
unscaled_data70 = data70s.iloc[:,0:-1]
unscaled_data80 = data80s.iloc[:,0:-1]
unscaled_data90 = data90s.iloc[:,0:-1]
unscaled_data00 = data00s.iloc[:,0:-1]
unscaled_data10 = data10s.iloc[:,0:-1]


target60 = data60s.iloc[:,[-1]] # all rows 
target70 = data70s.iloc[:,[-1]] # all rows 
target80 = data80s.iloc[:,[-1]] # all rows 
target90 = data90s.iloc[:,[-1]] # all rows 
target00 = data00s.iloc[:,[-1]] # all rows 
target10 = data10s.iloc[:,[-1]] # all rows 


# In[14]:


# Split dataset into train, validate and test sets
num_samples60 = unscaled_data60.shape[0]
num_samples70 = unscaled_data70.shape[0]
num_samples80 = unscaled_data80.shape[0]
num_samples90 = unscaled_data90.shape[0]
num_samples00 = unscaled_data00.shape[0]
num_samples10 = unscaled_data10.shape[0]


num_train_samples60 = int(0.8*num_samples60)
num_train_samples70 = int(0.8*num_samples70)
num_train_samples80 = int(0.8*num_samples80)
num_train_samples90 = int(0.8*num_samples90)
num_train_samples00 = int(0.8*num_samples00)
num_train_samples10 = int(0.8*num_samples10)


num_validation_samples60 = int(0.1*num_samples60) # for more accuracy
num_validation_samples70 = int(0.1*num_samples70) # for more accuracy
num_validation_samples80 = int(0.1*num_samples80) # for more accuracy
num_validation_samples90 = int(0.1*num_samples90) # for more accuracy
num_validation_samples00 = int(0.1*num_samples00) # for more accuracy
num_validation_samples10 = int(0.1*num_samples10) # for more accuracy


num_test_samples60 = num_samples60 - num_train_samples60 - num_validation_samples60
num_test_samples70 = num_samples70 - num_train_samples70 - num_validation_samples70
num_test_samples80 = num_samples80 - num_train_samples80 - num_validation_samples80
num_test_samples90 = num_samples90 - num_train_samples90 - num_validation_samples90
num_test_samples00 = num_samples00 - num_train_samples00 - num_validation_samples00
num_test_samples10 = num_samples10 - num_train_samples10 - num_validation_samples10


# In[15]:


# Train data set 
train_predictors60 = unscaled_data60[:num_train_samples60]
train_predictors70 = unscaled_data70[:num_train_samples70]
train_predictors80 = unscaled_data80[:num_train_samples80]
train_predictors90 = unscaled_data90[:num_train_samples90]
train_predictors00 = unscaled_data00[:num_train_samples00]
train_predictors10 = unscaled_data10[:num_train_samples10]



train_targets60 = target60[:num_train_samples60]
train_targets70 = target70[:num_train_samples70]
train_targets80 = target80[:num_train_samples80]
train_targets90 = target90[:num_train_samples90]
train_targets00 = target00[:num_train_samples00]
train_targets10 = target10[:num_train_samples10]


# In[16]:


# Test data set
test_predictors60 = unscaled_data60[num_train_samples60+num_validation_samples60:]
test_predictors70 = unscaled_data70[num_train_samples70+num_validation_samples70:]
test_predictors80 = unscaled_data80[num_train_samples80+num_validation_samples80:]
test_predictors90 = unscaled_data90[num_train_samples90+num_validation_samples90:]
test_predictors00 = unscaled_data00[num_train_samples00+num_validation_samples00:]
test_predictors10 = unscaled_data10[num_train_samples10+num_validation_samples10:]

test_targets60 = target60[num_train_samples60+num_validation_samples60:]
test_targets70 = target70[num_train_samples70+num_validation_samples70:]
test_targets80 = target80[num_train_samples80+num_validation_samples80:]
test_targets90 = target90[num_train_samples90+num_validation_samples90:]
test_targets00 = target00[num_train_samples00+num_validation_samples00:]
test_targets10 = target10[num_train_samples10+num_validation_samples10:]


# In[17]:


# Standardize predictor training data
scaler = StandardScaler()

x_train60 = scaler.fit_transform(train_predictors60)
x_test60 = scaler.transform(test_predictors60)

x_train70 = scaler.fit_transform(train_predictors70)
x_test70 = scaler.transform(test_predictors70)

x_train80 = scaler.fit_transform(train_predictors80)
x_test80 = scaler.transform(test_predictors80)

x_train90 = scaler.fit_transform(train_predictors90)
x_test90 = scaler.transform(test_predictors90)

x_train00 = scaler.fit_transform(train_predictors00)
x_test00 = scaler.transform(test_predictors00)

x_train10 = scaler.fit_transform(train_predictors10)
x_test10 = scaler.transform(test_predictors10)


# ### Evaluation of Logistic Regression Models

# In[18]:


# Define function to calculate LR model evaluation metrics
def evaluate_model(test_targets, y_pred, year):
    p_score = precision_score(test_targets, y_pred)
    r_score = recall_score(test_targets, y_pred)
    f1 = f1_score(test_targets, y_pred)
    cfmatrix = confusion_matrix(test_targets, y_pred)
    print('For ' , year, ': ', 'Confusion matrix: ', cfmatrix, ', Precision score: ', p_score, ', R Score:  ', r_score, ', F1 score: ', f1)
    


# In[19]:


# Training data model accuracy
def test_accuracy(model):
    print("Here are the accuracy scores for all decades: \n")
    trainaccuracyscores = [(model.score(x_train60, train_targets60)),
                       (model.score(x_train70, train_targets70)),
                        (model.score(x_train80, train_targets80)),
                         (model.score(x_train90, train_targets90)),
                          (model.score(x_train00, train_targets00)),
                           (model.score(x_train10, train_targets10))]

    trainaccuracyscores=pd.DataFrame(trainaccuracyscores,index=[1960,1970,1980,1990,2000,2010])
    trainaccuracyscores.columns=['\n LR Model Training Data Accuracy Scores']
    print(trainaccuracyscores)
    
    # Test data model accuracy
    testaccuracyscores = [(model.score(x_test60, test_targets60)),
                       (model.score(x_test70, test_targets70)),
                        (model.score(x_test80, test_targets80)),
                         (model.score(x_test90, test_targets90)),
                          (model.score(x_test00, test_targets00)),
                           (model.score(x_test10, test_targets10))]
    testaccuracyscores=pd.DataFrame(testaccuracyscores,index=[1960,1970,1980,1990,2000,2010])
    testaccuracyscores.columns=['\n LR Model Test Data Accuracy Scores \n']
    print(testaccuracyscores)
    
    y_pred60 = model.predict(x_test60)
    evaluate_model(test_targets60, y_pred60, '1960') #evaluate 1960s model
    y_pred70 = model.predict(x_test70)
    evaluate_model(test_targets70, y_pred70, '1970')
    y_pred80 = model.predict(x_test80)
    evaluate_model(test_targets80, y_pred80, '1980')
    y_pred90 = model.predict(x_test90)
    evaluate_model(test_targets90, y_pred90, '1990')
    y_pred00 = model.predict(x_test00)
    evaluate_model(test_targets00, y_pred00, '2000')
    y_pred10 = model.predict(x_test10)
    evaluate_model(test_targets10, y_pred10, '2010')


# # Similar Song Recommendations Generator
# ### KNN model to create a playlist based on user song choice

# In[20]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[21]:


#Import data of all decades
data60s = pd.read_csv('dataset-of-60s.csv')
data70s = pd.read_csv('dataset-of-70s.csv')
data80s = pd.read_csv('dataset-of-80s.csv')
data90s = pd.read_csv('dataset-of-90s.csv')
data00s = pd.read_csv('dataset-of-00s.csv')
data10s = pd.read_csv('dataset-of-10s.csv')


# In[22]:


#Add column to each dataset identify decade of data
data60s['decade']=1960
data70s['decade']=1970
data80s['decade']=1980
data90s['decade']=1990
data00s['decade']=2000
data10s['decade']=2010


# In[23]:


#Merge 6 decade datasets into one single dataframe
merged_data = pd.concat([data60s,data70s,data80s,data90s,data00s,data10s],axis=0)
merged_data


# In[24]:


#Convert decade to date-time format
merged_data['decade']=pd.to_datetime(merged_data['decade'], format='%Y')

#Drop non-numeric columns 
merged_data2=merged_data.drop(columns=['track','artist','uri','decade'])

#Convert dataframe to list of lists where one row of dataframe is a list entry
merged_data_list=merged_data2.values.tolist()
merged_data_list

# Input Spotify dataset
data = merged_data_list


# In[25]:


#Import Spotify packages
import os
os.environ['SPOTIPY_CLIENT_ID'] = '12a274d91fb54d6f95f9ab4589be1d48'
os.environ['SPOTIPY_CLIENT_SECRET'] ='1adab3171d2a4dab99e4114f462a965d'


# In[26]:


#Import Spotify web API tools
import spotipy
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="12a274d91fb54d6f95f9ab4589be1d48",
                                                           client_secret="1adab3171d2a4dab99e4114f462a965d"))


# Nearest neighbor derivation functions source from: #https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# 

# In[27]:


#Create list of song recommendations based on matching criteria
def create_list(song_recommendations):
    mylist = []

    for i in range(0,len(song_recommendations)):
        mylist.append(merged_data.index[(merged_data['danceability'] == song_recommendations[i][0]) & (merged_data['energy'] == song_recommendations[i][1])
        & (merged_data['key'] == song_recommendations[i][2]) & (merged_data['loudness'] == song_recommendations[i][3])   
        & (merged_data['mode'] == song_recommendations[i][4]) & (merged_data['speechiness'] == song_recommendations[i][5])
        & (merged_data['acousticness'] == song_recommendations[i][6]) & (merged_data['instrumentalness'] == song_recommendations[i][7])        
        & (merged_data['liveness'] == song_recommendations[i][8]) & (merged_data['valence'] == song_recommendations[i][9])                   
        & (merged_data['tempo'] == song_recommendations[i][10]) & (merged_data['duration_ms'] == song_recommendations[i][11])
        & (merged_data['time_signature'] == song_recommendations[i][12]) & (merged_data['chorus_hit'] == song_recommendations[i][13])                   
        & (merged_data['sections'] == song_recommendations[i][14]).tolist()])
    
    #Return playlist of k song recommendations to user 
    print('\nCreating playlist of similar songs...\n')
    df = pd.DataFrame()
    for i in range(0,len(mylist)):
        df=df.append(merged_data.iloc[mylist[i]])
    df.reset_index(inplace=True)
    df.drop(columns=['index'],inplace=True)

    print(df[['artist','track']])


# In[28]:


# Euclidean distance between two songs
from math import sqrt

def prox(song1, song2):
   dist = 0.0
   for i in range(len(song1)-1):
       dist += (song1[i] - song2[i])**2
   return sqrt(dist)

# Identify nearest neighbor songs by Euclidean distance of numerical data features
def find_recs(dataset, input_song, num_recs):
   distances = list()
   for song_entry in dataset:
       dist = prox(input_song, song_entry)
       distances.append((song_entry, dist))
   distances.sort(key=lambda tup: tup[1])
   recommendations = list()
   for i in range(num_recs):
       recommendations.append(distances[i][0])
   create_list(recommendations)


# In[29]:


def k_entry(songchoice):
    i=0
    while i ==0:
        #Select number of song recommendations preferred
        k_input=int(input('\nPlease enter how many recommendations you want: '))
        # Input new song with 16 numerical features
        if k_input>0:
            find_recs(data, songchoice, k_input)            
            i=1
        elif k_input<=0:
            print("Please enter a positive number.")


# In[30]:


#Extract songs from Spotify API

def get_features2(track_link):
    featurelist = sp.audio_features(tracks=[track_id])
    ftvalues = featurelist[0].values()
    ftvalues_list = list(ftvalues)

    #Convert dictionary's values to a list
    dnce = ftvalues_list[0]
    enrg = ftvalues_list[1]
    ky = ftvalues_list [2]
    ldns = ftvalues_list[3]
    mde = ftvalues_list[4]
    spch = ftvalues_list[5]
    acst = ftvalues_list[6]
    instr = ftvalues_list[7]
    livne = ftvalues_list[8]
    val = ftvalues_list[9]
    tmp = ftvalues_list[10]
    dur = ftvalues_list[16]
    ts = ftvalues_list[17]    
    songfeatures = (dnce, enrg, ky, ldns, mde, spch, acst,instr, livne, val,tmp, dur, ts) 
    k_entry(songfeatures) # call next function


# In[31]:


# function to choose song
def songchoosing():
    songnamelist = []
    chosentrack = input('\nInput the name of a song by your chosen artist on Spotify: ') 

    results = sp.search(q=chosentrack, limit=1)
    for track in results['tracks']['items']:
        track_id = track['uri']
        print('\nYour Song Choice: ', track['name'])
        songnamelist.append(chosentrack)
        if not songnamelist:
            print("\nPlease enter a valid song name by your chosen artist.")
        else:
            satisfied = input('\n Are you satisfied with ' + track['name'] + ' as your song selection? [Y/N]\n') 
            get_features2(track_id) # call next function
            while satisfied == 'N':
                songchoosing()


# In[32]:


# function for user to decide if they want to use the kNN function or end the service here
def goToRec():
    userinput = input("\nWould you like to find similar songs to this song? (Y/N)")
    if userinput in ('Y', 'y'): 
        print("\nWelcome to our song recommender! This service is useful if you desire to find another song that gives you the same magical feeling of the song you input!")
        songchoosing()
    else: 
        print('\n Thank you for using our service! Have a great day.')


# #### Import Spotify API tools

# In[33]:


### Spotify API requests using spotipy module
import os
os.environ['SPOTIPY_CLIENT_ID'] = '12a274d91fb54d6f95f9ab4589be1d48'
os.environ['SPOTIPY_CLIENT_SECRET'] ='1adab3171d2a4dab99e4114f462a965d'

import spotipy
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="12a274d91fb54d6f95f9ab4589be1d48",
                                                           client_secret="1adab3171d2a4dab99e4114f462a965d"))


# In[34]:


# Create function to train and predict with LR model
def buildModel(train,label,songfeatures, yearinput):
    model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
    model.fit(train, label)
    # call function to evaluate here
    x_test_new = scaler.transform(songfeatures)
    #Make prediction of hit or not
    hit_pred = model.predict(x_test_new)
    #Return hit or no hit response to user
    if hit_pred == 0: 
        print ('\nSorry! Your song would not be a hit in the ' + str(yearinput) + 's.')
        ask = input('Would you like to try a different year? (Y/N)')
        if ask in ('Y', 'y'): 
            yearinput=input("Choose a decade (1960, 1970, 1980, 1990, 2000, or 2010) to see if your song would be a hit or not in that decade: \n\n")
            getYear(yearinput, songfeatures)
        else:
            accuracytest = input('Would you like to test the accuracy of this result? (Y/N)')
            if accuracytest in ('Y', 'y'):
                print('\ntesting accuracy of the model....')
                test_accuracy(model)
                goToRec()
            else:
                goToRec()
    else: 
        print ('\nCongratulations! Your song would be a hit in the ' + str(yearinput) + 's.')
        ask = input('Would you like to try a different year? (Y/N)')
        if ask in ('Y', 'y'): 
            yearinput=input("Choose a decade (1960, 1970, 1980, 1990, 2000, or 2010) to see if your song would be a hit or not in that decade: \n\n")
            getYear(yearinput, songfeatures)
        else:
            accuracytest = input('Would you like to test the accuracy of our prediction model? (Y/N)')
            if accuracytest in ('Y', 'y'):
                print('\ntesting accuracy of the model....')
                test_accuracy(model)
                goToRec()
            else:
                goToRec()


# In[35]:


# Prompt user for decade of choice to base song hit evaluation off of
def getYear(yearinput, songfeatures):
    songfeatures=np.array(songfeatures)
    songfeatures=songfeatures.reshape(1, -1)
        # Run LR model for year chosen
    if yearinput == '1960':
        buildModel(x_train60,train_targets60, songfeatures, yearinput)
    elif yearinput == '1970':
        buildModel(x_train70,train_targets70, songfeatures, yearinput)
    elif yearinput == '1980':
        buildModel(x_train80,train_targets80, songfeatures, yearinput)
    elif yearinput == '1990':
        buildModel(x_train90,train_targets90, songfeatures, yearinput)
    elif yearinput == '2000':
        buildModel(x_train00,train_targets00, songfeatures, yearinput)
    elif yearinput == '2010':
        buildModel(x_train10,train_targets10, songfeatures, yearinput)
    else: 
        print('\nCannot recognize year. Try again')
        yearinput=input("Choose a decade (1960, 1970, 1980, 1990, 2000, or 2010) to see if the song would be a hit or not in that decade: \n\n")
        getYear(yearinput, songfeatures)


# In[36]:


# function to graph song on plot of danceability and loudness, and show user where inputted 
def graph_song(songfeatures):
    import plotly.express as px
    dnce = songfeatures[0]
    ldns = songfeatures[3]
    df2=pd.DataFrame({'dnce':[dnce],'ldns':[ldns]})
    df = merged_data
    fig1 = px.scatter(df, x='danceability', y='loudness', size='valence', size_max = 30, color = 'target', opacity = .30, hover_data=['decade', 'mode'], title = 'Hit vs Flop, where Yellow = Hit')
    fig1.show()
    fig2 = px.scatter(df, x='danceability', y='loudness', size='valence', size_max = 30, color = 'target', opacity = .01, hover_data=['decade', 'mode'], title = "Where your Song Falls on the Hit vs Flop Chart")
    # fig2.add_scatter(x = [0.4], y=[-10], size_max = 80, mode="markers")
    yourfig = fig2.add_scatter(x = df2['dnce'], y = df2['ldns'], mode="markers",
                    marker=dict(size=40, color="Red"),
                    name="Your Song Here")
    yourfig.show()
    yearinput=input("Choose a decade (1960, 1970, 1980, 1990, 2000, or 2010) to see if the song would be a hit or not in that decade: \n\n")
    getYear(yearinput, songfeatures)


# In[37]:


# Function to pull features from songs on Spotify
def get_features(track_id):
    featurelist = sp.audio_features(tracks=[track_id])
    ftvalues = featurelist[0].values()
    ftvalues_list = list(ftvalues)

    #Convert dictionary's values to a list
    dnce = ftvalues_list[0]
    enrg = ftvalues_list[1]
    ky = ftvalues_list [2]
    ldns = ftvalues_list[3]
    mde = ftvalues_list[4]
    spch = ftvalues_list[5]
    acst = ftvalues_list[6]
    instr = ftvalues_list[7]
    livne = ftvalues_list[8]
    val = ftvalues_list[9]
    tmp = ftvalues_list[10]
    dur = ftvalues_list[16]
    ts = ftvalues_list[17]
    songfeatures = [dnce, enrg, ky, ldns, mde, spch, acst, instr, livne, val, tmp, dur, ts]
    graph_song(songfeatures)


# In[ ]:


# the final, only code that run - all inputs will be prompted with this block. 
print("Hello! Welcome to the Song Hit Predictor! This program allows you to input any song on Spotify's database, and we'll tell you if your song would've been a hit or not in a certain decade of your choosing.\n")

searchorenter = input('Would you like to search for your song or enter a link? (S or L)')
if searchorenter in ('S','s'): 
    artistquery = input('Enter artist name to search for track: ')

    results = sp.search(q=artistquery, limit=10)
    for idx, track in enumerate(results['tracks']['items']):
        print(idx, track['name'])


    chosentrack = input('\nInput your song choice: ') 

    results = sp.search(q=chosentrack, limit=1)
    for idx, track in enumerate(results['tracks']['items']):
        track_id = track['uri']
        print('\nYou chose this song: ', track['name'])
        print('Analyzing and comparing "' + track['name'] + '" in our database of hits vs flops..... \n')
        get_features(track_id)

elif searchorenter in ('L', 'l'):
    url = input("\nEnter a link to your song: ")
    track = sp.track(url)
    print('\nYou chose this song: ' +  track['name'])
    print('Analyzing and comparing "' + track['name'] + '" in our database of hits vs flops..... \n')
    uri = track['uri']
    get_features(uri)
else: 
    print("\nPlease re-run code and enter S or L")

