#!/usr/bin/env python
# coding: utf-8

# # Part C: Graphic User Interface
# #### Using tKinter, this builds a GUI allowing the user to type in songs and click buttons to more quickly get their result. 

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


# this data needs to be downloaded - see package 
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


# #### Logistic Regression Model Building

# In[5]:


# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[6]:


# Upload all decade song data files
data60s = pd.read_csv('dataset-of-60s.csv')
data70s = pd.read_csv('dataset-of-70s.csv')
data80s = pd.read_csv('dataset-of-80s.csv')
data90s = pd.read_csv('dataset-of-90s.csv')
data00s = pd.read_csv('dataset-of-00s.csv')
data10s = pd.read_csv('dataset-of-10s.csv')


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


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


# #### Import Spotify API tools

# In[12]:


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


# In[13]:


# Create function to train and predict with LR model
def buildModel(train,label,songfeatures, yearinput):
    model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
    model.fit(train, label)
    
    # call function to evaluate here
    x_test_new = scaler.transform(songfeatures)
    
    #make prediction of hit or not
    hit_pred = model.predict(x_test_new)
    
    #Return hit or no hit response to user
    if hit_pred == 0: 
        txt =  'Sorry! This song would not be a hit in the ' + str(yearinput) + 's.'
        hitlabel['text'] = txt
        hitlabel['fg'] = 'red'
    else: 
        txt = 'Congratulations! This song would be a hit in the ' + str(yearinput) + 's.'
        hitlabel['text'] = txt
        hitlabel['fg'] = 'green'


# In[14]:


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
    
    features = [dnce, enrg, ky, ldns, mde, spch, acst, instr, livne, val, tmp, dur, ts]
    songfeatures = np.array(features)
    songfeatures = songfeatures.reshape(1, -1)
    
    # buttons for decades - can click on as many times as you want
    button1960 = tk.Button(root,
                              relief='sunken',
                              bd=2,
                              text='1960',
                              font=('Futura Medium', 15),
                              justify='center',
                              state='active',
                              command=lambda: buildModel(x_train60,train_targets60, songfeatures, 1960))
    button1960.place(relx=0.45, relwidth=0.05, rely=0.3, relheight=0.03, anchor='n')

    button1970 = tk.Button(root,
                              relief='sunken',
                              bd=2,
                              text='1970',
                              font=('Futura Medium', 15),
                              justify='center',
                              state='active',
                              command=lambda: buildModel(x_train70,train_targets70, songfeatures, 1970))
    button1970.place(relx=0.5, relwidth=0.05, rely=0.3, relheight=0.03, anchor='n')

    button1980 = tk.Button(root,
                              relief='sunken',
                              bd=2,
                              text='1980',
                              font=('Futura Medium', 15),
                              justify='center',
                              state='active',
                              command=lambda: buildModel(x_train80,train_targets80, songfeatures, 1980))
    button1980.place(relx=0.55, relwidth=0.05, rely=0.3, relheight=0.03, anchor='n')

    button1990 = tk.Button(root,
                              relief='sunken',
                              bd=2,
                              text='1990',
                              font=('Futura Medium', 15),
                              justify='center',
                              state='active',
                              command=lambda: buildModel(x_train90,train_targets90, songfeatures, 1990))
    button1990.place(relx=0.6, relwidth=0.05, rely=0.3, relheight=0.03, anchor='n')

    button2000 = tk.Button(root,
                              relief='sunken',
                              bd=2,
                              text='2000',
                              font=('Futura Medium', 15),
                              justify='center',
                              state='active',
                              command=lambda: buildModel(x_train00,train_targets00, songfeatures, 2000))
    button2000.place(relx=0.65, relwidth=0.05, rely=0.3, relheight=0.03, anchor='n')

    button2001 = tk.Button(root,
                              relief='sunken',
                              bd=2,
                              text='2001',
                              font=('Futura Medium', 15),
                              justify='center',
                              state='active',
                              command=lambda: buildModel(x_train10,train_targets10, songfeatures, 2010))
    button2001.place(relx=0.7, relwidth=0.05, rely=0.3, relheight=0.03, anchor='n')


# # Tkinter

# In[15]:


# TKINTER 
import tkinter as tk # needs pip install
import webbrowser
from PIL import ImageTk, Image # needs pip install

# building the canvas 
root = tk.Tk()
root.title('Music Moirai')
canvas = tk.Canvas(root, height=1000, width=1500)
canvas.pack()

#setting background 
img = tk.PhotoImage (file = 'music2.gif' , master = root) 
background_label = tk.Label(root, image = img, bg = 'black')
background_label.place(relwidth=1, relheight=1)

# song enter field
nameEntry = tk.StringVar()
enterName = tk.Entry(root,
                     font=('Futura', 15),
                     bg = 'black', 
                     fg = 'white',
                     bd = 3,
                     textvariable=nameEntry,
                     justify='center')
enterName.place(relx=0.07, relwidth=0.1, rely=0.1, relheight=0.03, anchor='n')

nameButton = tk.Button(root,
                       height=1,
                       width=10,
                       text="Enter Name",
                       font=('Futura Medium', 15),
                       command=lambda: getUserName(enterName.get()))
nameButton.place(relx=0.17, relwidth=0.1, rely=0.1, relheight=0.03, anchor='n')

# about button
introButton = tk.Button(root,
                        height=1,
                        width=10,
                        text="About Music Morai",
                        font=('Futura Medium', 13),
                        command=lambda: introDisplay())
introButton.place(relx=0.5, relwidth=0.1, rely=0.8, relheight=0.03, anchor='n')

# song entry field
frame = tk.Frame(root, bg='black')  # bd = border
frame.place(relx=0.5, relwidth=0.5, rely=0.2, relheight=0.05, anchor='n')
entry = tk.StringVar()
songEntry = tk.Entry(frame,
                       bg='black',
                       fg = 'white',
                       font=('Futura', 14))  # text entry field
songEntry.place(relwidth=0.65, relheight=1)  # 1 means fill it

# FIRST BUTTON - submit link
linkbutton = tk.Button(frame,
                   text='Predict Demo',
                   font=('Futura Medium', 15),
                   bg ='black',
                   fg = 'black',
                   command=lambda: enterLink(songEntry.get()))
linkbutton.place(relx=0.6, relwidth=0.2, relheight=1)


# to open Spotify app to find URL
def callback(url):
    webbrowser.open_new(url)

spoturl = 'https://open.spotify.com/'

linkbutton = tk.Button(frame,
                   text='Go to Spotify',
                   font=('Futura Medium', 15),
                   command=lambda: callback(spoturl))
linkbutton.place(relx=0.8, relwidth=0.2, relheight=1)


# RETURN FIELD - HIT OR NOT
lower_frame = tk.Frame(root, bg='black')
lower_frame.place(relx=0.5, rely=0.35, relwidth=0.4, relheight=0.05, anchor='n')
# x and y are positions
hitlabel = tk.Label(lower_frame,
                 bg='black',
                 font=('Futura Medium', 25),
                 anchor='n',
                 justify='center',
                 bd=4)
hitlabel.place(relwidth=1, relheight=1)

# to get more info about what music moirai is about (button)
def introDisplay():
    top = tk.Toplevel(bg='#80c1ff', bd=2, height=400, width=400)
    top.title("About Us...")
    about_message = 'Welcome to Music Moirai! We are a service for record labels to reduce time from the tedious '                     'act of listening to thousands of demos each day. Moirai means "destiny", and we provide you with '                     'insight into whether the demo you input is a hit or not, saving you tons of time and work '                     'for your A&Rs. '
    msg = tk.Message(top, text=about_message)
    msg.place(relwidth=1, relheight=1)
    button = tk.Button(top, text="Dismiss", font=('Futura', 10), command=top.destroy)
    button.place(relx=0.5, rely=0.75)
    
# function to get user's name (make it more personable)
def getUserName(text):
    nameEntry.set(text)
    nameLabel = tk.Label(root, text='Welcome to the Music Moirai, ' + text + '! Enter a demo link \nbelow from a song on Spotify to predict if the song will be a hit or not.',
                         font=('Futura', 15), bg = 'black', fg = 'white')
    nameLabel.place(relx=0.5, relwidth=0.5, rely=0.11, relheight=0.05, anchor='n')
    nameButton['state'] = 'disabled'
    
# user to proceed button
GetRecsButton = tk.Button(root,
                          relief='sunken',
                          bd=2,
                          text='Predict!  ' + entry.get(),
                          font=('Futura Medium', 15),
                          justify='center',
                          state='disabled',
                          command=lambda: get_features(songEntry.get()))
GetRecsButton.place(relx=0.5, relwidth=0.5, rely=0.25, relheight=0.04, anchor='n')

# when user clicks 'predict demo'
def enterLink(url):
    if len(url) < 30: 
        GetRecsButton['text'] = 'Invalid URL. Try Again.'
    track = sp.track(url)
    uri = track['uri']
    GetRecsButton['state'] = 'active'
    GetRecsButton['text'] = '\nYou chose "' +  track['name'] + '" by ' + track['artists'][0]['name'] + '. Click to analyze in our database of hits vs flops.. \n'
    
# when user clicks 'search for demo'
def enterArtist(artist):
    button['state'] = 'disabled'
    GetRecsButton['state'] = 'active'
                       
decadeLabel = tk.Label(root, text='Choose a decade to test the demo', bg = 'black', fg = 'white', font=('Futura Medium', 12))
decadeLabel.place(relx=0.33, relwidth=0.15, rely=0.3, relheight=0.03, anchor='n')

# welcome label 
welcomeLabel = tk.Label(root, text='Music Moirai: Test your Song\'s Destiny', bg = 'black', fg = 'white', font=('Futura Medium', 25))
welcomeLabel.place(relx=0.5, relwidth=0.3, rely=0.05, relheight=0.04, anchor='n')


# In[16]:


# main function - the only block that runs and outputs the entire GUI
root = tk.mainloop()


# In[ ]:




