# Music Morai
## Machine Learning Music Predictor Model
**Music Morai** is a machine learning music popularity predictor model for labels to streamline their process and efficiently select artists. It predicts which artists or songs will become popular based on a song's attributes. 
<br></br> 

## Goal
Our project aims to assist the A&R (Artist and Repertoire) team that finds new artists to sign to the label and help the record label understand the nuances of not only what creates a “hit,” but a long-lasting crowd favorite. 

By analyzing the top 2,000 songs of all time on Spotify and identifying trends for upcoming artists, it would be possible to predict and identify artists who will release songs that will be hits in the United States.

Specifically, we investigated on:
- What indicators (e.g. BPM, key, danceability, energy scores) lead to higher favorability amongst music listeners? 
- Is there a perfect recipe/formula for a song to be a massive hit?
- What elements of a song lend itself to a longer lasting popularity throughout the years?
<br></br> 

## Demo
<p align="center">
  <img width="900" src="doc/CBI_Project_Demo.gif">
</p>

## Instruction
1. Search your song by entering either the artist name or the URL from Spotify.
2. Choose a song from the search result by entering the song name.
3. Choose a time period in which you would like to see if the song would be a hit or not.
4. Based on the trained dataset through ML, the model will indicate whether the selected song would have been hit or not. 
<br></br>

## Tech + Libraries
- **Python** (Jupyter)
    - **pandas**
    - **NumPy**
- Machine Learning
    - **scikit-learn**
- Data Visualization 
    - **Matplotlib**
    - **seaborn**
- API Connection 
    - **ssl**
    - **spotipy**
<br></br>

## Context of Dataset
**[Spotify - All Time Top 2000s Mega Dataset](https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset)** is a dataset collected from **[Kaggle](https://www.kaggle.com/)** which we used to train and build our model. It consists of features for tracks fetched using Spotify's Web API. The tracks are labeled '1' or '0' ('Hit' or 'Flop') depending on some criterias of the author.This dataset can be used to make a classification model that predicts whether a track would be a 'Hit' or not.

(Note: The author does not objectively considers a track inferior, bad or a failure if its labeled 'Flop'. 'Flop' here merely implies that it is not a song that probably could not be considered popular in the mainstream.)
<br></br>

## Attributes
For further reading: https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/

- **Track**: The Name of the track.

- **Artist**: The Name of the Artist.

- **URI**: The resource identifier for the track.

- **Danceability**: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 

- **Energy**: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. 

- **Key**: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C?/D?, 2 = D, and so on. If no key was detected, the value is -1.

- **Loudness**: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. 
        
- **Mode**: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

- speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. 

- **Acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. The distribution of values for this feature look like this:

- **Instrumentalness**: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. The distribution of values for this feature look like this:

- **Liveness**: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

- **Valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

- **Tempo**: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. 

- **Duration_ms**: 	The duration of the track in milliseconds.

- **Time_signature**: An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).

- **Chorus_hit**: This the the author's best estimate of when the chorus would start for the track. Its the timestamp of the start of the third section of the track (in milliseconds). This feature was extracted from the data recieved by the API call for Audio Analysis of that particular track.

- **Sections**: The number of sections the particular track has. This feature was extracted from the data recieved by the API call for Audio Analysis of that particular track.

- **Target**: The target variable for the track. It can be either '0' or '1'. '1' implies that this song has featured in the weekly list (Issued by Billboards) of Hot-100 tracks in that decade at least once and is therefore a 'hit'. '0' Implies that the track is a 'flop'.
<br></br>

The author's condition of a track being 'flop' is as follows:

- The track must not appear in the 'hit' list of that decade.
- The track's artist must not appear in the 'hit' list of that decade.
- The track must belong to a genre that could be considered non-mainstream and / or avant-garde. 
- The track's genre must not have a song in the 'hit' list.
- The genre list for the particular decades are as follows:
- The track must have 'US' as one of its markets.
<br></br>

## For More Details of The Project
- **[Project Presentation](https://github.com/myeon7/Music-Morai/blob/main/doc/CBI_Project_Presentation.pdf)**
- **[Project Report](https://github.com/myeon7/Music-Morai/blob/main/doc/CBI_Project_Report.pdf)**
<br></br>

## Acknowledgement
- **spotipy**: Python module for Spotify's API (https://pypi.org/project/spotipy/)
- **billboard**: Python module for Billboard's API (https://pypi.org/project/billboard.py/)
- Spotify, the company itself. For keeping a database of such in-depth details of every track in their library. And for exposing their API for the world to use.