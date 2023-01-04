import streamlit as st
from PIL import Image
import subprocess
import sys
import time
import os
import random
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu as om
import re
from ContentTFIDF import ContentTFIDF
from ContentBasedRecommender import ContentBasedRecommender

def cleanText(readData):
    text = re.sub('[-=+#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', readData)
 
    return text

def done_selecting():
    return False

def add_favorite_track(track):
    if track:
        st.session_state["favorite_tracks"].append(track)

def transform_user_starting_preferences(mood, speed, emotion):
    if mood == "light":
        st.session_state["user_mood"] = 1
    else:
        st.session_state["user_mood"] = 2

    if speed == "fast":
        st.session_state["user_speed"] = 1
    else:
        st.session_state["user_speed"] = 2

    if emotion == "happy":
        st.session_state["user_emotion"] = 1
    elif emotion == "meh":
        st.session_state["user_emotion"] = 2
    else:
        st.session_state["user_emotion"] = 3
        


if "favorite_tracks" not in st.session_state:
    st.session_state["favorite_tracks"] = []
    st.session_state["user_mood"] = None
    st.session_state["user_speed"] = None
    st.session_state["user_emotion"] = None

tracks = pd.read_csv('tracks.csv')
tracks = pd.DataFrame(tracks)

genres = []
for i in tracks['artist_genre']:
    if i == '[]':
        i = 'NA'
        genres.append(i.strip())
    else:
        i = cleanText(i)
        genres.append(i.strip())
tracks['genre'] = genres

tracks = tracks[tracks['genre'] != "NA"]
tracks = tracks.reset_index()
tracks['track_popularity'] = tracks['track_popularity'] / 100

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    initial_sidebar_state="collapsed",
)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

selected = om("Main menu", ['Music Recommender 🎵', 'Statistics 📊'], icons=['upload','list-task'], menu_icon='cast', default_index=0, orientation='horizontal')

st.title = "Music Recommender 🎵"

if selected == "Music Recommender 🎵":
    st.header("Music recommender", anchor="title")
    st.write("Let's get to know you better")
    st.subheader("1. Select the one of the following", anchor="remove-cold-start")

    artists = tracks['artist_name'].sort_values().tolist()

    genres.insert(0, "")
    artists.insert(0, "")
    col1,col2 = st.columns(2)
    genre = col1.selectbox('Filter by genre:', dict.fromkeys(genres), index=0, key="genre-selector")
    artist = col2.selectbox('Filter by artist:', dict.fromkeys(artists), index=0, key="artist-selector")

    if genre != "" and artist != "":
        filtered_tracks = tracks.loc[(tracks["genre"] == genre) & (tracks["artist_name"] == artist)]
    else:
        if genre != "" or artist != "":
            filtered_tracks = tracks.loc[(tracks["genre"] == genre) | (tracks["artist_name"] == artist)]
        else:
            filtered_tracks = tracks

    filtered_tracks = filtered_tracks["track_name"].tolist()
    filtered_tracks.insert(0, "")
    selecting_tracks = True
    selected_track = st.selectbox('Select your favorite tracks', filtered_tracks, key="track-selector")

    add_favorite_track(selected_track)

    # add_song_btn.button('Add song', on_click=add_favorite_track(selected_track, True), key="add-song-btn")
    selecting_tracks = st.button('Done', on_click=done_selecting, key="done-adding-songs-btn")

    st.write(st.session_state["favorite_tracks"])
    if selecting_tracks != False:
        st.success('Done')

    mood = st.selectbox("Light or dark song?", ["light", "dark"], key="light-dark-selectbox")
    speed = st.selectbox("Fast songs or slow songs", ["fast", "slow"], key="fast-slow-selectbox")
    emotion = st.selectbox("What's your current mood?", ["happy", "meh", "sad"], key="mood-selectbox")

    selecting_moods = True
    selecting_moods = st.button('Done', on_click=done_selecting, key="done-choosing-mood-btn")

    if selecting_moods != False:
        transform_user_starting_preferences(mood, speed, emotion)

        ct = ContentTFIDF(tracks)
        ct_tfidf = ct.calculateTFIDF()

        if st.session_state["user_mood"] != None and st.session_state["user_speed"] != None and st.session_state["user_emotion"] != None and len(st.session_state["favorite_tracks"]) > 0:
            content_recommender = ContentBasedRecommender(tracks, ct_tfidf, st.session_state["favorite_tracks"], st.session_state["user_mood"], st.session_state["user_speed"], st.session_state["user_emotion"])
            content_recommender.recommend_features()

elif selected == "Statistics 📊":
    st.header("Statistics", anchor="statistics")