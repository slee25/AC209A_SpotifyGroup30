---
title: Data Collection and EDA
notebook: notebooks/Data_Collection_EDA.ipynb
nav_include: 1
---


## Contents
{:.no_toc}
*  
{: toc}


```python
import warnings

import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

import json

from bs4 import BeautifulSoup
import requests
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS


# Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
client_credentials_manager = SpotifyClientCredentials(client_id='e6ff82a6418a4191a5b3a95622faf5dd', client_secret='a37b632dc07d4136902fa95ec56281d3')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Genius
Genius_TOKEN = 'C2ow8dBpT2W5ORhiqaiz8ht8zLs9UzjFJQS5fwsmkRwWZyj8Wi1dA37FXYjScYuu'

import re

pd.set_option('display.width', 1500) 
pd.set_option('display.max_columns', 100)
```


## 1. Research on Available Databases

### (1) Million Playlist Dataset

- Million Playlist Dataset is a very large playlist dataset provided by Spotify, particularly prepared for RecSys Challenge 2018 (https://recsys-challenge.spotify.com/). It consists of 1000 files in .json format, and each file contains 1000 user-created playlists, which makes 1 million playlists in 5.4 GB. The dataset was generated in December, 2017.


- Data format:
    - For each playlist, available data are:
        - `name` (string): The name of the playlist.
        - `collaborative` (Boolean): `true` if the owner allows other users to modify the playlist.
        - `pid` (int): Position ID number of the playlist in the dataset (0 - 999999).
        - `modified_at` (timestamp): The timestamp when the playlist was list modified.
        - `num_tracks` (int): The number of tracks in the playlist.
        - `num_albums` (int): The number of unique albums in the playlist.
        - `num_followers` (int): The number of followers of the playlist.
        - `tracks` (dictionary): The list of tracks in the playlist (see below).
        - `num_edits` (int): The number of edits that has been made so far.
        - `duration_ms` (int): The total duration of the playlist in milliseconds.
        - `num_artists` (int): The number of unique artists in the playlist.
    - For each track in a playlist, available data are:
        - `pos` (int): Position ID number of the track in the playlist.
        - `artist_name` (string): The name of the artist of the track.
        - `track_uri` (uri): The URI for the track.
        - `artist_uri` (uri): The URI for the artist of the track.
        - `track_name` (string): The name of the track.
        - `album_uri` (uri): The URI for the album of the track.
        - `duration_ms` (int): The duration of the track in milliseconds.
        - `album_name` (string): The name of the album of the track.


- Below shows an example data format for the 0th playlist in the 0th file:



```python
FILENAME = 'data/mpd.slice.0-999.json'
with open(FILENAME, "r") as fd:
    playlist_data = json.load(fd)
    
playlist_data['playlists'][0]
```





    {'name': 'Throwbacks',
     'collaborative': 'false',
     'pid': 0,
     'modified_at': 1493424000,
     'num_tracks': 52,
     'num_albums': 47,
     'num_followers': 1,
     'tracks': [{'pos': 0,
       'artist_name': 'Missy Elliott',
       'track_uri': 'spotify:track:0UaMYEvWZi0ZqiDOoHU3YI',
       'artist_uri': 'spotify:artist:2wIVse2owClT7go1WT98tk',
       'track_name': 'Lose Control (feat. Ciara & Fat Man Scoop)',
       'album_uri': 'spotify:album:6vV5UrXcfyQD1wu4Qo2I9K',
       'duration_ms': 226863,
       'album_name': 'The Cookbook'},
      {'pos': 1,
       'artist_name': 'Britney Spears',
       'track_uri': 'spotify:track:6I9VzXrHxO9rA9A5euc8Ak',
       'artist_uri': 'spotify:artist:26dSoYclwsYLMAKD3tpOr4',
       'track_name': 'Toxic',
       'album_uri': 'spotify:album:0z7pVBGOD7HCIB7S8eLkLI',
       'duration_ms': 198800,
       'album_name': 'In The Zone'},
      {'pos': 2,
       'artist_name': 'Beyoncé',
       'track_uri': 'spotify:track:0WqIKmW4BTrj3eJFmnCKMv',
       'artist_uri': 'spotify:artist:6vWDO969PvNqNYHIOW5v0m',
       'track_name': 'Crazy In Love',
       'album_uri': 'spotify:album:25hVFAxTlDvXbx2X2QkUkE',
       'duration_ms': 235933,
       'album_name': 'Dangerously In Love (Alben für die Ewigkeit)'},
      {'pos': 3,
       'artist_name': 'Justin Timberlake',
       'track_uri': 'spotify:track:1AWQoqb9bSvzTjaLralEkT',
       'artist_uri': 'spotify:artist:31TPClRtHm23RisEBtV3X7',
       'track_name': 'Rock Your Body',
       'album_uri': 'spotify:album:6QPkyl04rXwTGlGlcYaRoW',
       'duration_ms': 267266,
       'album_name': 'Justified'},
      {'pos': 4,
       'artist_name': 'Shaggy',
       'track_uri': 'spotify:track:1lzr43nnXAijIGYnCT8M8H',
       'artist_uri': 'spotify:artist:5EvFsr3kj42KNv97ZEnqij',
       'track_name': "It Wasn't Me",
       'album_uri': 'spotify:album:6NmFmPX56pcLBOFMhIiKvF',
       'duration_ms': 227600,
       'album_name': 'Hot Shot'},
      {'pos': 5,
       'artist_name': 'Usher',
       'track_uri': 'spotify:track:0XUfyU2QviPAs6bxSpXYG4',
       'artist_uri': 'spotify:artist:23zg3TcAtWQy7J6upgbUnj',
       'track_name': 'Yeah!',
       'album_uri': 'spotify:album:0vO0b1AvY49CPQyVisJLj0',
       'duration_ms': 250373,
       'album_name': 'Confessions'},
      {'pos': 6,
       'artist_name': 'Usher',
       'track_uri': 'spotify:track:68vgtRHr7iZHpzGpon6Jlo',
       'artist_uri': 'spotify:artist:23zg3TcAtWQy7J6upgbUnj',
       'track_name': 'My Boo',
       'album_uri': 'spotify:album:1RM6MGv6bcl6NrAG8PGoZk',
       'duration_ms': 223440,
       'album_name': 'Confessions'},
      {'pos': 7,
       'artist_name': 'The Pussycat Dolls',
       'track_uri': 'spotify:track:3BxWKCI06eQ5Od8TY2JBeA',
       'artist_uri': 'spotify:artist:6wPhSqRtPu1UhRCDX5yaDJ',
       'track_name': 'Buttons',
       'album_uri': 'spotify:album:5x8e8UcCeOgrOzSnDGuPye',
       'duration_ms': 225560,
       'album_name': 'PCD'},
      {'pos': 8,
       'artist_name': "Destiny's Child",
       'track_uri': 'spotify:track:7H6ev70Weq6DdpZyyTmUXk',
       'artist_uri': 'spotify:artist:1Y8cdNmUJH7yBTd9yOvr5i',
       'track_name': 'Say My Name',
       'album_uri': 'spotify:album:283NWqNsCA9GwVHrJk59CG',
       'duration_ms': 271333,
       'album_name': "The Writing's On The Wall"},
      {'pos': 9,
       'artist_name': 'OutKast',
       'track_uri': 'spotify:track:2PpruBYCo4H7WOBJ7Q2EwM',
       'artist_uri': 'spotify:artist:1G9G7WwrXka3Z1r7aIDjI7',
       'track_name': 'Hey Ya! - Radio Mix / Club Mix',
       'album_uri': 'spotify:album:1UsmQ3bpJTyK6ygoOOjG1r',
       'duration_ms': 235213,
       'album_name': 'Speakerboxxx/The Love Below'},
      {'pos': 10,
       'artist_name': 'Nelly Furtado',
       'track_uri': 'spotify:track:2gam98EZKrF9XuOkU13ApN',
       'artist_uri': 'spotify:artist:2jw70GZXlAI8QzWeY2bgRc',
       'track_name': 'Promiscuous',
       'album_uri': 'spotify:album:2yboV2QBcVGEhcRlYuPpDT',
       'duration_ms': 242293,
       'album_name': 'Loose'},
      {'pos': 11,
       'artist_name': 'Jesse McCartney',
       'track_uri': 'spotify:track:4Y45aqo9QMa57rDsAJv40A',
       'artist_uri': 'spotify:artist:2Hjj68yyUPiC0HKEOigcEp',
       'track_name': 'Right Where You Want Me - Radio Edit Version',
       'album_uri': 'spotify:album:6022khQj4Fsvvse8f3A4lF',
       'duration_ms': 211693,
       'album_name': 'Right Where You Want Me'},
      {'pos': 12,
       'artist_name': 'Jesse McCartney',
       'track_uri': 'spotify:track:1HwpWwa6bnqqRhK8agG4RS',
       'artist_uri': 'spotify:artist:2Hjj68yyUPiC0HKEOigcEp',
       'track_name': 'Beautiful Soul',
       'album_uri': 'spotify:album:2gidE8vgVOkYufANjuvj3S',
       'duration_ms': 214226,
       'album_name': 'Beautiful Soul'},
      {'pos': 13,
       'artist_name': 'Jesse McCartney',
       'track_uri': 'spotify:track:20ORwCJusz4KS2PbTPVNKo',
       'artist_uri': 'spotify:artist:2Hjj68yyUPiC0HKEOigcEp',
       'track_name': "Leavin'",
       'album_uri': 'spotify:album:2tDCfKFy2YW9N0IwNSRdOJ',
       'duration_ms': 216880,
       'album_name': 'Departure - Recharged'},
      {'pos': 14,
       'artist_name': 'Cassie',
       'track_uri': 'spotify:track:7k6IzwMGpxnRghE7YosnXT',
       'artist_uri': 'spotify:artist:27FGXRNruFoOdf1vP8dqcH',
       'track_name': 'Me & U',
       'album_uri': 'spotify:album:0j1qzjaJmsF1FkcICf3hRu',
       'duration_ms': 192213,
       'album_name': 'Cassie'},
      {'pos': 15,
       'artist_name': 'Omarion',
       'track_uri': 'spotify:track:1Bv0Yl01xBDZD4OQP93fyl',
       'artist_uri': 'spotify:artist:0f5nVCcR06GX8Qikz0COtT',
       'track_name': 'Ice Box',
       'album_uri': 'spotify:album:4cVVEOXyUaWo7vlDWIkKsI',
       'duration_ms': 256426,
       'album_name': '21'},
      {'pos': 16,
       'artist_name': 'Avril Lavigne',
       'track_uri': 'spotify:track:4omisSlTk6Dsq2iQD7MA07',
       'artist_uri': 'spotify:artist:0p4nmQO2msCgU4IF37Wi3j',
       'track_name': 'Sk8er Boi',
       'album_uri': 'spotify:album:7h6XeTzy0SRXDrFJeA9gO7',
       'duration_ms': 204000,
       'album_name': 'Let Go'},
      {'pos': 17,
       'artist_name': 'Chris Brown',
       'track_uri': 'spotify:track:7xYnUQigPoIDAMPVK79NEq',
       'artist_uri': 'spotify:artist:7bXgB6jMjp9ATFy66eO08Z',
       'track_name': 'Run It!',
       'album_uri': 'spotify:album:49gaz5rhWWgqCw61M9700v',
       'duration_ms': 229866,
       'album_name': 'Chris Brown'},
      {'pos': 18,
       'artist_name': 'Beyoncé',
       'track_uri': 'spotify:track:6d8A5sAx9TfdeseDvfWNHd',
       'artist_uri': 'spotify:artist:6vWDO969PvNqNYHIOW5v0m',
       'track_name': 'Check On It - feat. Bun B and Slim Thug',
       'album_uri': 'spotify:album:3MJHoQUI828kmB6IpjejbW',
       'duration_ms': 210453,
       'album_name': "B'Day"},
      {'pos': 19,
       'artist_name': "Destiny's Child",
       'track_uri': 'spotify:track:4pmc2AxSEq6g7hPVlJCPyP',
       'artist_uri': 'spotify:artist:1Y8cdNmUJH7yBTd9yOvr5i',
       'track_name': "Jumpin', Jumpin'",
       'album_uri': 'spotify:album:283NWqNsCA9GwVHrJk59CG',
       'duration_ms': 230200,
       'album_name': "The Writing's On The Wall"},
      {'pos': 20,
       'artist_name': 'Sheryl Crow',
       'track_uri': 'spotify:track:215JYyyUnrJ98NK3KEwu6d',
       'artist_uri': 'spotify:artist:4TKTii6gnOnUXQHyuo9JaD',
       'track_name': 'Soak Up The Sun',
       'album_uri': 'spotify:album:5NYcTXrRZHxNyRKVOd0vs1',
       'duration_ms': 292306,
       'album_name': "C'Mon C'Mon"},
      {'pos': 21,
       'artist_name': 'The Black Eyed Peas',
       'track_uri': 'spotify:track:0uqPG793dkDDN7sCUJJIVC',
       'artist_uri': 'spotify:artist:1yxSLGMDHlW21z4YXirZDS',
       'track_name': 'Where Is The Love?',
       'album_uri': 'spotify:album:1bNyYpkDRovmErm4QeDrpJ',
       'duration_ms': 272533,
       'album_name': 'Elephunk'},
      {'pos': 22,
       'artist_name': 'Bowling For Soup',
       'track_uri': 'spotify:track:19Js5ypV6JKn4DMExHQbGc',
       'artist_uri': 'spotify:artist:5ND0mGcL9SKSjWIjPd0xIb',
       'track_name': "Stacy's Mom",
       'album_uri': 'spotify:album:3Q7xpHmP8k3HryE0LQdIk0',
       'duration_ms': 193042,
       'album_name': "I've Never Done Anything Like This"},
      {'pos': 23,
       'artist_name': 'The Click Five',
       'track_uri': 'spotify:track:1JURww012QnWAw0zZXi6Aa',
       'artist_uri': 'spotify:artist:01lz5VBfkMFDteSA9pKJuP',
       'track_name': 'Just The Girl',
       'album_uri': 'spotify:album:7gZilZGYr8M7UwEeYvdAKZ',
       'duration_ms': 234146,
       'album_name': 'Greetings From Imrie House'},
      {'pos': 24,
       'artist_name': 'Chris Brown',
       'track_uri': 'spotify:track:7DFnq8FYhHMCylykf6ZCxA',
       'artist_uri': 'spotify:artist:7bXgB6jMjp9ATFy66eO08Z',
       'track_name': 'Yo (Excuse Me Miss)',
       'album_uri': 'spotify:album:49gaz5rhWWgqCw61M9700v',
       'duration_ms': 229040,
       'album_name': 'Chris Brown'},
      {'pos': 25,
       'artist_name': 'Jonas Brothers',
       'track_uri': 'spotify:track:1TfAhjzRBWzYZ8IdUV3igl',
       'artist_uri': 'spotify:artist:7gOdHgIoIKoe4i9Tta6qdD',
       'track_name': 'Year 3000',
       'album_uri': 'spotify:album:20RAjvZ9LX2FDuDU8RDuIl',
       'duration_ms': 201960,
       'album_name': 'Jonas Brothers'},
      {'pos': 26,
       'artist_name': 'Lil Mama',
       'track_uri': 'spotify:track:1Y4ZdPOOgCUhBcKZOrUFiS',
       'artist_uri': 'spotify:artist:5qK5bOC6wLtuLhG5KvU17c',
       'track_name': 'Lip Gloss',
       'album_uri': 'spotify:album:3vgVsm9GY3i39fZ7b1sqV5',
       'duration_ms': 219773,
       'album_name': 'Lip Gloss'},
      {'pos': 27,
       'artist_name': 'Cascada',
       'track_uri': 'spotify:track:6MjljecHzHelUDismyKkba',
       'artist_uri': 'spotify:artist:0N0d3kjwdY2h7UVuTdJGfp',
       'track_name': 'Everytime We Touch - Radio Edit',
       'album_uri': 'spotify:album:5DvuKZTzEKjm0oUuhP237C',
       'duration_ms': 199120,
       'album_name': 'Everytime We Touch'},
      {'pos': 28,
       'artist_name': 'Jason Derulo',
       'track_uri': 'spotify:track:67T6l4q3zVjC5nZZPXByU8',
       'artist_uri': 'spotify:artist:07YZf4WDAMNwqr4jfgOZ8y',
       'track_name': 'Whatcha Say',
       'album_uri': 'spotify:album:0aVJmVAeEx78nAA1rAKYf7',
       'duration_ms': 221253,
       'album_name': 'Jason Derulo'},
      {'pos': 29,
       'artist_name': 'Ne-Yo',
       'track_uri': 'spotify:track:34ceTg8ChN5HjrqiIYCn9Q',
       'artist_uri': 'spotify:artist:21E3waRsmPlU7jZsS13rcj',
       'track_name': 'Miss Independent',
       'album_uri': 'spotify:album:6dTn9vJSxVTIGm4Cu5dH4x',
       'duration_ms': 232000,
       'album_name': 'Year Of The Gentleman'},
      {'pos': 30,
       'artist_name': 'Miley Cyrus',
       'track_uri': 'spotify:track:5Q0Nhxo0l2bP3pNjpGJwV1',
       'artist_uri': 'spotify:artist:5YGY8feqx7naU7z4HrwZM6',
       'track_name': 'Party In The U.S.A.',
       'album_uri': 'spotify:album:64aKkqxc3Ur2LYIKeS5osS',
       'duration_ms': 202066,
       'album_name': 'The Time Of Our Lives'},
      {'pos': 31,
       'artist_name': 'Boys Like Girls',
       'track_uri': 'spotify:track:6GIrIt2M39wEGwjCQjGChX',
       'artist_uri': 'spotify:artist:0vWCyXMrrvMlCcepuOJaGI',
       'track_name': 'The Great Escape',
       'album_uri': 'spotify:album:4WqgusSAgXkrjbXzqdBY68',
       'duration_ms': 206520,
       'album_name': 'Boys Like Girls'},
      {'pos': 32,
       'artist_name': 'Iyaz',
       'track_uri': 'spotify:track:4E5P1XyAFtrjpiIxkydly4',
       'artist_uri': 'spotify:artist:5tKXB9uuebKE34yowVaU3C',
       'track_name': 'Replay',
       'album_uri': 'spotify:album:44hyrGuZKAvITbmrlhryf8',
       'duration_ms': 182306,
       'album_name': 'Replay'},
      {'pos': 33,
       'artist_name': 'Chris Brown',
       'track_uri': 'spotify:track:3H1LCvO3fVsK2HPguhbml0',
       'artist_uri': 'spotify:artist:7bXgB6jMjp9ATFy66eO08Z',
       'track_name': 'Forever',
       'album_uri': 'spotify:album:1UtE4zAlSE2TlKmTFgrTg5',
       'duration_ms': 277106,
       'album_name': 'Exclusive - The Forever Edition'},
      {'pos': 34,
       'artist_name': 'Kesha',
       'track_uri': 'spotify:track:3uoQULcUWfnt6nc6J7Vgai',
       'artist_uri': 'spotify:artist:6LqNN22kT3074XbTVUrhzX',
       'track_name': 'Your Love Is My Drug',
       'album_uri': 'spotify:album:5peRwC6pQh8eaoIPtvmmOB',
       'duration_ms': 187133,
       'album_name': 'Animal'},
      {'pos': 35,
       'artist_name': 'Ne-Yo',
       'track_uri': 'spotify:track:2nbClS09zsIAqNkshg6jnp',
       'artist_uri': 'spotify:artist:21E3waRsmPlU7jZsS13rcj',
       'track_name': 'Closer',
       'album_uri': 'spotify:album:1nv3KEXZPmcwOXMoLTs1vn',
       'duration_ms': 234360,
       'album_name': 'Year Of The Gentleman'},
      {'pos': 36,
       'artist_name': 'Justin Bieber',
       'track_uri': 'spotify:track:69ghzc538EQSVon2Gm3wrr',
       'artist_uri': 'spotify:artist:1uNFoZAHBGtllmzznpCI3s',
       'track_name': 'One Less Lonely Girl',
       'album_uri': 'spotify:album:1rG5TDs3jYh6OU753I54CI',
       'duration_ms': 229106,
       'album_name': 'My World'},
      {'pos': 37,
       'artist_name': 'M.I.A.',
       'track_uri': 'spotify:track:1kusepF3AacIEtUTYrw4GV',
       'artist_uri': 'spotify:artist:0QJIPDAEDILuo8AIq3pMuU',
       'track_name': 'Paper Planes',
       'album_uri': 'spotify:album:1Lymt1abGCr3J06bbnmWca',
       'duration_ms': 203760,
       'album_name': 'Kala'},
      {'pos': 38,
       'artist_name': 'The Killers',
       'track_uri': 'spotify:track:7oK9VyNzrYvRFo7nQEYkWN',
       'artist_uri': 'spotify:artist:0C0XlULifJtAgn6ZNCW2eu',
       'track_name': 'Mr. Brightside',
       'album_uri': 'spotify:album:4undIeGmofnAYKhnDclN1w',
       'duration_ms': 222586,
       'album_name': 'Hot Fuss'},
      {'pos': 39,
       'artist_name': 'blink-182',
       'track_uri': 'spotify:track:12qZHAeOyTf93YAWvGDTat',
       'artist_uri': 'spotify:artist:6FBDaR13swtiWwGhX1WQsP',
       'track_name': 'All The Small Things',
       'album_uri': 'spotify:album:1fF8kYX49s5Ufv4XEY5sjW',
       'duration_ms': 168000,
       'album_name': 'Enema Of The State'},
      {'pos': 40,
       'artist_name': 'The Pussycat Dolls',
       'track_uri': 'spotify:track:2jFlMILIQzs7lSFudG9lbo',
       'artist_uri': 'spotify:artist:6wPhSqRtPu1UhRCDX5yaDJ',
       'track_name': 'Beep',
       'album_uri': 'spotify:album:0ylxpXE00fVxh6d60tevT8',
       'duration_ms': 229360,
       'album_name': 'PCD'},
      {'pos': 41,
       'artist_name': 'Justin Bieber',
       'track_uri': 'spotify:track:4I2GqMe7L2ccMpUbnDzYLH',
       'artist_uri': 'spotify:artist:1uNFoZAHBGtllmzznpCI3s',
       'track_name': 'Somebody To Love',
       'album_uri': 'spotify:album:6gdLfnf2vdNlMTyhJHaDLs',
       'duration_ms': 220920,
       'album_name': 'My Worlds'},
      {'pos': 42,
       'artist_name': 'The All-American Rejects',
       'track_uri': 'spotify:track:5lDriBxJd22IhOH9zTcFrV',
       'artist_uri': 'spotify:artist:3vAaWhdBR38Q02ohXqaNHT',
       'track_name': 'Dirty Little Secret',
       'album_uri': 'spotify:album:3PWEGZ6CYvXRnr0JCECsDe',
       'duration_ms': 193653,
       'album_name': 'Move Along'},
      {'pos': 43,
       'artist_name': 'Justin Bieber',
       'track_uri': 'spotify:track:2eJ8ij1T3cNUKiGdcUvKhy',
       'artist_uri': 'spotify:artist:1uNFoZAHBGtllmzznpCI3s',
       'track_name': 'Baby',
       'album_uri': 'spotify:album:6gdLfnf2vdNlMTyhJHaDLs',
       'duration_ms': 213973,
       'album_name': 'My Worlds'},
      {'pos': 44,
       'artist_name': 'Vanessa Carlton',
       'track_uri': 'spotify:track:5y69gQtK33qxb8a24ACkCy',
       'artist_uri': 'spotify:artist:5ILrArfIV0tMURcHJN8Q07',
       'track_name': 'A Thousand Miles',
       'album_uri': 'spotify:album:7D6BFTArx2ajtkKRVXIKO2',
       'duration_ms': 237493,
       'album_name': 'Be Not Nobody'},
      {'pos': 45,
       'artist_name': 'Cris Cab',
       'track_uri': 'spotify:track:1X5WGCrUMuwRFuYU1eAo2I',
       'artist_uri': 'spotify:artist:7vWBZm3sQ8yQvfV4nXxHXK',
       'track_name': 'Livin on Sunday',
       'album_uri': 'spotify:album:2kNznk4KDkYXifzOAUDoXN',
       'duration_ms': 201230,
       'album_name': 'Red Road'},
      {'pos': 46,
       'artist_name': 'Miley Cyrus',
       'track_uri': 'spotify:track:3utIAb67sOu0QHxBE88P1M',
       'artist_uri': 'spotify:artist:5YGY8feqx7naU7z4HrwZM6',
       'track_name': 'See You Again',
       'album_uri': 'spotify:album:6SkirMQoL4QhnXOM5MH5El',
       'duration_ms': 190453,
       'album_name': 'See You Again'},
      {'pos': 47,
       'artist_name': 'Jesse McCartney',
       'track_uri': 'spotify:track:3jkdQNkDTxxXtjSO4l0o1H',
       'artist_uri': 'spotify:artist:2Hjj68yyUPiC0HKEOigcEp',
       'track_name': 'How Do You Sleep? - Featuring Ludacris',
       'album_uri': 'spotify:album:2tDCfKFy2YW9N0IwNSRdOJ',
       'duration_ms': 208333,
       'album_name': 'Departure - Recharged'},
      {'pos': 48,
       'artist_name': 'Demi Lovato',
       'track_uri': 'spotify:track:5c1sfI6wIQEsSUw0xrkFdl',
       'artist_uri': 'spotify:artist:6S2OmqARrzebs0tKUEyXyp',
       'track_name': 'This Is Me',
       'album_uri': 'spotify:album:6vykWEBzBfEKYJxEFR1AQl',
       'duration_ms': 189186,
       'album_name': 'Camp Rock Original Soundtrack'},
      {'pos': 49,
       'artist_name': 'Avril Lavigne',
       'track_uri': 'spotify:track:6sqNctd7MlJoKDOxPVCAvU',
       'artist_uri': 'spotify:artist:0p4nmQO2msCgU4IF37Wi3j',
       'track_name': 'My Happy Ending',
       'album_uri': 'spotify:album:7851Vsjv3apS52sXUik6iF',
       'duration_ms': 242413,
       'album_name': 'Under My Skin'},
      {'pos': 50,
       'artist_name': 'We The Kings',
       'track_uri': 'spotify:track:1b7vg5T9YKR3NNqXfBYRF7',
       'artist_uri': 'spotify:artist:3ao3jf5d70Tf4fPh2bnXVl',
       'track_name': 'Check Yes Juliet',
       'album_uri': 'spotify:album:2F1hfUOuMnOxtSfrktL8VX',
       'duration_ms': 220133,
       'album_name': 'We The Kings'},
      {'pos': 51,
       'artist_name': 'Boys Like Girls',
       'track_uri': 'spotify:track:6GIrIt2M39wEGwjCQjGChX',
       'artist_uri': 'spotify:artist:0vWCyXMrrvMlCcepuOJaGI',
       'track_name': 'The Great Escape',
       'album_uri': 'spotify:album:4WqgusSAgXkrjbXzqdBY68',
       'duration_ms': 206520,
       'album_name': 'Boys Like Girls'}],
     'num_edits': 6,
     'duration_ms': 11532414,
     'num_artists': 37}



- Notes:
    - As the entire dataset is too large, data resizing (resampling) is required.
    - Although not many types of data are available in the dataset, but it does provide URIs for track, artist, and album for each track, which can be used in Spotify API to query more data (see below).

### (2) Spotify Web API

- Spotify Web API (https://developer.spotify.com/documentation/web-api/) is a developer's tool provided by Spotify, particularly for developing web applications related to Spotify. Based on simple REST principles, the Spotify Web API endpoints return JSON metadata about music artists, albums, and tracks, directly from the Spotify Data Catalogue. Web API also provides access to user related data, like playlists and music that the user saves in the Your Music library. Such access is enabled through selective authorization, by the user. The API provides a set of endpoints, each with its own unique path. To access private data through the Web API, such as user profiles and playlists, an application must get the user's permission to access the data. Authorization is via the Spotify Accounts service.

<img src="https://developer.spotify.com/assets/WebAPI_intro.png" width="600" height="250">

- For this project, we have used Spotipy (https://spotipy.readthedocs.io/), which is a lightweight Python library for the Spotify Web API. With Spotipy we can get full access to all of the music data provided by the Spotify platform. Below we list some key function endpoints provided by Spotipy.

- `spotipy.Spotify().track(track_uri)`: Get Spotify catalog information for a single track identified by its unique Spotify ID. (https://developer.spotify.com/documentation/web-api/reference/tracks/get-track/)
    - `album` (a simplified `album object`): The album on which the track appears. The album object includes a link in `href` to full information about the album.
    - `artists` (an array of simplified `artist objects`): The artists who performed the track. Each artist object includes a link in `href` to more detailed information about the artist.
    - `available_markets` (array of strings): A list of the countries in which the track can be played, identified by their `ISO 3166-1 alpha-2 code`.
    - `disc_number` (int): 	The disc number (usually `1` unless the album consists of more than one disc).
    - `duration_ms` (int): The track length in milliseconds.
    - `explicit` (Boolean): Whether or not the track has explicit lyrics (`true` = yes it does; `false` = no it does not OR unknown).
    - `external_ids` (an `external ID object`): Known external IDs for the track.
    - `external_urls` (an `external URL object`): Known external URLs for this track.
    - `href` (string): A link to the Web API endpoint providing full details of the track.
    - `id` (string): The `Spotify ID` for the track.
    - `name` (string): The name of the track.
    - `popularity` (int): The popularity of the track. The value will be between 0 and 100, with 100 being the most popular. The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity. Note that the popularity value may lag actual popularity by a few days: the value is not updated in real time.
    - `preview_url` (string): A link to a 30 second preview (MP3 format) of the track. Can be `null`.
    - `track_number` (int): The number of the track. If an album has several discs, the track number is the number on the specified disc.
    - `type` (string): The object type, `track`.
    - `uri` (string): The `Spotify URI` for the track.
    
    
- Below shows an example data provided by `spotipy.Spotify().track(track_uri)` for the 0th track in the 0th playlist in the 0th file:



```python
TRACK_URI = playlist_data['playlists'][0]['tracks'][0]['track_uri']
sp.track(TRACK_URI)
```





    {'album': {'album_type': 'album',
      'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
        'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
        'id': '2wIVse2owClT7go1WT98tk',
        'name': 'Missy Elliott',
        'type': 'artist',
        'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
      'available_markets': ['AD',
       'AE',
       'AR',
       'AT',
       'AU',
       'BE',
       'BG',
       'BH',
       'BO',
       'BR',
       'CA',
       'CH',
       'CL',
       'CO',
       'CR',
       'CY',
       'CZ',
       'DE',
       'DK',
       'DO',
       'DZ',
       'EC',
       'EE',
       'EG',
       'ES',
       'FI',
       'FR',
       'GB',
       'GR',
       'GT',
       'HK',
       'HN',
       'HU',
       'ID',
       'IE',
       'IL',
       'IS',
       'IT',
       'JO',
       'JP',
       'KW',
       'LB',
       'LI',
       'LT',
       'LU',
       'LV',
       'MA',
       'MC',
       'MT',
       'MX',
       'MY',
       'NI',
       'NL',
       'NO',
       'NZ',
       'OM',
       'PA',
       'PE',
       'PH',
       'PL',
       'PS',
       'PT',
       'PY',
       'QA',
       'RO',
       'SA',
       'SE',
       'SG',
       'SK',
       'SV',
       'TH',
       'TN',
       'TR',
       'TW',
       'US',
       'UY',
       'VN',
       'ZA'],
      'external_urls': {'spotify': 'https://open.spotify.com/album/6vV5UrXcfyQD1wu4Qo2I9K'},
      'href': 'https://api.spotify.com/v1/albums/6vV5UrXcfyQD1wu4Qo2I9K',
      'id': '6vV5UrXcfyQD1wu4Qo2I9K',
      'images': [{'height': 640,
        'url': 'https://i.scdn.co/image/608d68c28beb4d3fb2891ba983965c28d550f592',
        'width': 640},
       {'height': 300,
        'url': 'https://i.scdn.co/image/facb5dc58151f589a8328d0aa148425e67e260ea',
        'width': 300},
       {'height': 64,
        'url': 'https://i.scdn.co/image/03b79b1f4336fe04ec3b62d9fd01f2034bff4fb5',
        'width': 64}],
      'name': 'The Cookbook',
      'release_date': '2005-07-04',
      'release_date_precision': 'day',
      'total_tracks': 16,
      'type': 'album',
      'uri': 'spotify:album:6vV5UrXcfyQD1wu4Qo2I9K'},
     'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
       'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
       'id': '2wIVse2owClT7go1WT98tk',
       'name': 'Missy Elliott',
       'type': 'artist',
       'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/2NdeV5rLm47xAvogXrYhJX'},
       'href': 'https://api.spotify.com/v1/artists/2NdeV5rLm47xAvogXrYhJX',
       'id': '2NdeV5rLm47xAvogXrYhJX',
       'name': 'Ciara',
       'type': 'artist',
       'uri': 'spotify:artist:2NdeV5rLm47xAvogXrYhJX'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/15GGbJKqC6w0VYyAJtjej6'},
       'href': 'https://api.spotify.com/v1/artists/15GGbJKqC6w0VYyAJtjej6',
       'id': '15GGbJKqC6w0VYyAJtjej6',
       'name': 'Fatman Scoop',
       'type': 'artist',
       'uri': 'spotify:artist:15GGbJKqC6w0VYyAJtjej6'}],
     'available_markets': ['AD',
      'AE',
      'AR',
      'AT',
      'AU',
      'BE',
      'BG',
      'BH',
      'BO',
      'BR',
      'CA',
      'CH',
      'CL',
      'CO',
      'CR',
      'CY',
      'CZ',
      'DE',
      'DK',
      'DO',
      'DZ',
      'EC',
      'EE',
      'EG',
      'ES',
      'FI',
      'FR',
      'GB',
      'GR',
      'GT',
      'HK',
      'HN',
      'HU',
      'ID',
      'IE',
      'IL',
      'IS',
      'IT',
      'JO',
      'JP',
      'KW',
      'LB',
      'LI',
      'LT',
      'LU',
      'LV',
      'MA',
      'MC',
      'MT',
      'MX',
      'MY',
      'NI',
      'NL',
      'NO',
      'NZ',
      'OM',
      'PA',
      'PE',
      'PH',
      'PL',
      'PS',
      'PT',
      'PY',
      'QA',
      'RO',
      'SA',
      'SE',
      'SG',
      'SK',
      'SV',
      'TH',
      'TN',
      'TR',
      'TW',
      'US',
      'UY',
      'VN',
      'ZA'],
     'disc_number': 1,
     'duration_ms': 226863,
     'explicit': True,
     'external_ids': {'isrc': 'USEE10414022'},
     'external_urls': {'spotify': 'https://open.spotify.com/track/0UaMYEvWZi0ZqiDOoHU3YI'},
     'href': 'https://api.spotify.com/v1/tracks/0UaMYEvWZi0ZqiDOoHU3YI',
     'id': '0UaMYEvWZi0ZqiDOoHU3YI',
     'is_local': False,
     'name': 'Lose Control (feat. Ciara & Fat Man Scoop)',
     'popularity': 62,
     'preview_url': 'https://p.scdn.co/mp3-preview/fe79cac4e5f67a23d54ea5e884d86af1b5248a88?cid=e6ff82a6418a4191a5b3a95622faf5dd',
     'track_number': 4,
     'type': 'track',
     'uri': 'spotify:track:0UaMYEvWZi0ZqiDOoHU3YI'}



- `spotipy.Spotify().artist(artist_uri)`: Get Spotify catalog information for a single artist identified by their unique Spotify ID. (https://developer.spotify.com/documentation/web-api/reference/artists/get-artist/)
    - `external_urls` (an `external URL object`): Known external URLs for this artist.
    - `followers` (a `followers object`): Information about the followers of the artist.
    - `genres` (array of strings): A list of the genres the artist is associated with. For example: `Prog Rock`, `Post-Grunge`. (If not yet classified, the array is empty.)
    - `href` (string): A link to the Web API endpoint providing full details of the artist.
    - `id` (string): The `Spotify ID` for the artist.
    - `images` (array of `image objects`): Images of the artist in various sizes, widest first.
    - `name` (string): The name of the artist.
    - `popularity` (int): The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist's popularity is calculated from the popularity of all the artist's tracks.
    - `type` (string): The object type, `artist`.
    - `uri` (string): The `Spotify URI` for the artist.


- Below shows an example data provided by `spotipy.Spotify().artist(artist_uri)` for the artist of the 0th track in the 0th playlist in the 0th file:



```python
ARTIST_URI = playlist_data['playlists'][0]['tracks'][0]['artist_uri']
sp.artist(ARTIST_URI)
```





    {'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
     'followers': {'href': None, 'total': 909586},
     'genres': ['dance pop',
      'hip hop',
      'hip pop',
      'pop',
      'pop rap',
      'r&b',
      'rap',
      'southern hip hop',
      'urban contemporary'],
     'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
     'id': '2wIVse2owClT7go1WT98tk',
     'images': [{'height': 640,
       'url': 'https://i.scdn.co/image/055260c034b93dae018b8cd70bc9f1acc2843af3',
       'width': 640},
      {'height': 320,
       'url': 'https://i.scdn.co/image/2642935f38deb4f2305cabfd996babec8796d469',
       'width': 320},
      {'height': 160,
       'url': 'https://i.scdn.co/image/11323b9db35fb2b10c1676a0eeeb5ff8a4ed32e8',
       'width': 160}],
     'name': 'Missy Elliott',
     'popularity': 76,
     'type': 'artist',
     'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}



- `spotipy.Spotify().album(album_uri)`: Get Spotify catalog information for a single album. (https://developer.spotify.com/documentation/web-api/reference/albums/get-album/)
    - `album_type` (string): The type of the album, one of `album`, `single`, or `compilation`.
    - `artists` (array of simplified `artist objects`): The artists of the album. Each artist object includes a link in `href` to more detailed information about the artist.
    - `available_markets` (array of strings): The markets in which the album is available in `ISO 3166-1 alpha-2 country codes`. Note that an album is considered available in a market when at least 1 of its tracks is available in that market.
    - `copyrights` (array of copyright objects): The copyright statements of the album.
    - `external_ids` (an `external ID object`): Known external IDs for the album.
    - `external_urls` (an `external URL object`): Known external URLs for this album.
    - `followers` (a `followers object`): Information about the followers of the artist.
    - `genres` (array of strings): A list of the genres used to classify the album. For example: `Prog Rock`, `Post-Grunge`. (If not yet classified, the array is empty.)
    - `href` (string): A link to the Web API endpoint providing full details of the album.
    - `id` (string): The `Spotify ID` for the album.
    - `images` (array of `image objects`): The cover art for the album in various sizes, widest first.
    - `label` (string): The label for the album.
    - `name` (string): The name of the album. In case of an album takedown, the value may be an empty string.
    - `popularity` (int): The popularity of the album. The value will be between 0 and 100, with 100 being the most popular. The popularity is calculated from the popularity of the album’s individual tracks.
    - `release_date` (string): The date the album was first released, for example `1981-12-15`. Depending on the precision, it might be shown as `1981` or `1981-12`.
    - `release_date_precision` (string): The precision with which release_date value is known, `year`, `month`, or `day`.
    - `tracks` (array of simplified `track objects` inside a `paging object`): The tracks of the album.
    - `type` (string): The object type, `album`.
    - `uri` (string): The `Spotify URI` for the album.


- Below shows an example data provided by `spotipy.Spotify().album(album_uri)` for the album of the 0th track in the 0th playlist in the 0th file:



```python
ALBUM_URI = playlist_data['playlists'][0]['tracks'][0]['album_uri']
sp.album(ALBUM_URI)
```





    {'album_type': 'album',
     'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
       'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
       'id': '2wIVse2owClT7go1WT98tk',
       'name': 'Missy Elliott',
       'type': 'artist',
       'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
     'available_markets': ['AD',
      'AE',
      'AR',
      'AT',
      'AU',
      'BE',
      'BG',
      'BH',
      'BO',
      'BR',
      'CA',
      'CH',
      'CL',
      'CO',
      'CR',
      'CY',
      'CZ',
      'DE',
      'DK',
      'DO',
      'DZ',
      'EC',
      'EE',
      'EG',
      'ES',
      'FI',
      'FR',
      'GB',
      'GR',
      'GT',
      'HK',
      'HN',
      'HU',
      'ID',
      'IE',
      'IL',
      'IS',
      'IT',
      'JO',
      'JP',
      'KW',
      'LB',
      'LI',
      'LT',
      'LU',
      'LV',
      'MA',
      'MC',
      'MT',
      'MX',
      'MY',
      'NI',
      'NL',
      'NO',
      'NZ',
      'OM',
      'PA',
      'PE',
      'PH',
      'PL',
      'PS',
      'PT',
      'PY',
      'QA',
      'RO',
      'SA',
      'SE',
      'SG',
      'SK',
      'SV',
      'TH',
      'TN',
      'TR',
      'TW',
      'US',
      'UY',
      'VN',
      'ZA'],
     'copyrights': [{'text': '2005 Atlantic Recording Corporation for the United States and WEA International Inc. for the world outside of the United States.',
       'type': 'C'},
      {'text': '2005 Atlantic Recording Corporation for the United States and WEA International Inc. for the world outside of the United States',
       'type': 'P'}],
     'external_ids': {'upc': '075678377969'},
     'external_urls': {'spotify': 'https://open.spotify.com/album/6vV5UrXcfyQD1wu4Qo2I9K'},
     'genres': [],
     'href': 'https://api.spotify.com/v1/albums/6vV5UrXcfyQD1wu4Qo2I9K',
     'id': '6vV5UrXcfyQD1wu4Qo2I9K',
     'images': [{'height': 640,
       'url': 'https://i.scdn.co/image/608d68c28beb4d3fb2891ba983965c28d550f592',
       'width': 640},
      {'height': 300,
       'url': 'https://i.scdn.co/image/facb5dc58151f589a8328d0aa148425e67e260ea',
       'width': 300},
      {'height': 64,
       'url': 'https://i.scdn.co/image/03b79b1f4336fe04ec3b62d9fd01f2034bff4fb5',
       'width': 64}],
     'label': 'Atlantic Records/ATG',
     'name': 'The Cookbook',
     'popularity': 61,
     'release_date': '2005-07-04',
     'release_date_precision': 'day',
     'total_tracks': 16,
     'tracks': {'href': 'https://api.spotify.com/v1/albums/6vV5UrXcfyQD1wu4Qo2I9K/tracks?offset=0&limit=50',
      'items': [{'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/07VmOvmuBp9G0gb8BTrpn0'},
          'href': 'https://api.spotify.com/v1/artists/07VmOvmuBp9G0gb8BTrpn0',
          'id': '07VmOvmuBp9G0gb8BTrpn0',
          'name': 'Mike Jones',
          'type': 'artist',
          'uri': 'spotify:artist:07VmOvmuBp9G0gb8BTrpn0'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 289773,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/5emRlAm3hfUrpPvdNLNXG0'},
        'href': 'https://api.spotify.com/v1/tracks/5emRlAm3hfUrpPvdNLNXG0',
        'id': '5emRlAm3hfUrpPvdNLNXG0',
        'is_local': False,
        'name': 'Joy (feat. Mike Jones)',
        'preview_url': 'https://p.scdn.co/mp3-preview/d90ce9a75b00634de5f6ec83b0e8f2e083e1372a?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 1,
        'type': 'track',
        'uri': 'spotify:track:5emRlAm3hfUrpPvdNLNXG0'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 184160,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/3dmNwPAKvydfjwpY3G1DVG'},
        'href': 'https://api.spotify.com/v1/tracks/3dmNwPAKvydfjwpY3G1DVG',
        'id': '3dmNwPAKvydfjwpY3G1DVG',
        'is_local': False,
        'name': 'Partytime',
        'preview_url': 'https://p.scdn.co/mp3-preview/cadbdda2c9689f5196ba7a1253d310e6e9156200?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 2,
        'type': 'track',
        'uri': 'spotify:track:3dmNwPAKvydfjwpY3G1DVG'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/1W9qOBYRTfP7HcizWN43G1'},
          'href': 'https://api.spotify.com/v1/artists/1W9qOBYRTfP7HcizWN43G1',
          'id': '1W9qOBYRTfP7HcizWN43G1',
          'name': 'Slick Rick',
          'type': 'artist',
          'uri': 'spotify:artist:1W9qOBYRTfP7HcizWN43G1'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 255853,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/5BTKwu4JTtMwTqPCXwEL1S'},
        'href': 'https://api.spotify.com/v1/tracks/5BTKwu4JTtMwTqPCXwEL1S',
        'id': '5BTKwu4JTtMwTqPCXwEL1S',
        'is_local': False,
        'name': 'Irresistible Delicious (feat. Slick Rick)',
        'preview_url': 'https://p.scdn.co/mp3-preview/84523df48fd70d5f6405453279aebbf305a10d4b?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 3,
        'type': 'track',
        'uri': 'spotify:track:5BTKwu4JTtMwTqPCXwEL1S'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/2NdeV5rLm47xAvogXrYhJX'},
          'href': 'https://api.spotify.com/v1/artists/2NdeV5rLm47xAvogXrYhJX',
          'id': '2NdeV5rLm47xAvogXrYhJX',
          'name': 'Ciara',
          'type': 'artist',
          'uri': 'spotify:artist:2NdeV5rLm47xAvogXrYhJX'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/15GGbJKqC6w0VYyAJtjej6'},
          'href': 'https://api.spotify.com/v1/artists/15GGbJKqC6w0VYyAJtjej6',
          'id': '15GGbJKqC6w0VYyAJtjej6',
          'name': 'Fatman Scoop',
          'type': 'artist',
          'uri': 'spotify:artist:15GGbJKqC6w0VYyAJtjej6'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 226863,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/0UaMYEvWZi0ZqiDOoHU3YI'},
        'href': 'https://api.spotify.com/v1/tracks/0UaMYEvWZi0ZqiDOoHU3YI',
        'id': '0UaMYEvWZi0ZqiDOoHU3YI',
        'is_local': False,
        'name': 'Lose Control (feat. Ciara & Fat Man Scoop)',
        'preview_url': 'https://p.scdn.co/mp3-preview/fe79cac4e5f67a23d54ea5e884d86af1b5248a88?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 4,
        'type': 'track',
        'uri': 'spotify:track:0UaMYEvWZi0ZqiDOoHU3YI'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/1XkoF8ryArs86LZvFOkbyr'},
          'href': 'https://api.spotify.com/v1/artists/1XkoF8ryArs86LZvFOkbyr',
          'id': '1XkoF8ryArs86LZvFOkbyr',
          'name': 'Mary J. Blige',
          'type': 'artist',
          'uri': 'spotify:artist:1XkoF8ryArs86LZvFOkbyr'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/6IjhOxJSTPh15KgFTSZ68K'},
          'href': 'https://api.spotify.com/v1/artists/6IjhOxJSTPh15KgFTSZ68K',
          'id': '6IjhOxJSTPh15KgFTSZ68K',
          'name': 'Grand Puba',
          'type': 'artist',
          'uri': 'spotify:artist:6IjhOxJSTPh15KgFTSZ68K'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 172186,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/3q15vMERXG9Sd4HcjxtQ46'},
        'href': 'https://api.spotify.com/v1/tracks/3q15vMERXG9Sd4HcjxtQ46',
        'id': '3q15vMERXG9Sd4HcjxtQ46',
        'is_local': False,
        'name': 'My Struggles (feat. Mary J. Blige & Grand Puba)',
        'preview_url': 'https://p.scdn.co/mp3-preview/44a24116e3ef89a133db4c953e847678c85e7748?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 5,
        'type': 'track',
        'uri': 'spotify:track:3q15vMERXG9Sd4HcjxtQ46'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 256546,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/3YqgabBYii4YaybPLnzpE4'},
        'href': 'https://api.spotify.com/v1/tracks/3YqgabBYii4YaybPLnzpE4',
        'id': '3YqgabBYii4YaybPLnzpE4',
        'is_local': False,
        'name': 'Meltdown',
        'preview_url': 'https://p.scdn.co/mp3-preview/d4cb2e86558662399490433068d2217198cacc1d?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 6,
        'type': 'track',
        'uri': 'spotify:track:3YqgabBYii4YaybPLnzpE4'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 285760,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/7aJouq94UPaX7yVXd2MQ4k'},
        'href': 'https://api.spotify.com/v1/tracks/7aJouq94UPaX7yVXd2MQ4k',
        'id': '7aJouq94UPaX7yVXd2MQ4k',
        'is_local': False,
        'name': 'On & On',
        'preview_url': 'https://p.scdn.co/mp3-preview/9df6d65e12388b04db66599ad24ed6d9ea6079e2?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 7,
        'type': 'track',
        'uri': 'spotify:track:7aJouq94UPaX7yVXd2MQ4k'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 204461,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/4z5fkIflIBvSG9elVNmiOJ'},
        'href': 'https://api.spotify.com/v1/tracks/4z5fkIflIBvSG9elVNmiOJ',
        'id': '4z5fkIflIBvSG9elVNmiOJ',
        'is_local': False,
        'name': 'We Run This',
        'preview_url': 'https://p.scdn.co/mp3-preview/9a48425bf46069f37cd72ab4797f2414ef6e054a?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 8,
        'type': 'track',
        'uri': 'spotify:track:4z5fkIflIBvSG9elVNmiOJ'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 258813,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/6tX2z3DnwUGTBgctckRDYs'},
        'href': 'https://api.spotify.com/v1/tracks/6tX2z3DnwUGTBgctckRDYs',
        'id': '6tX2z3DnwUGTBgctckRDYs',
        'is_local': False,
        'name': 'Remember When',
        'preview_url': 'https://p.scdn.co/mp3-preview/932fad7fd97b3182fb60d14fed6aa183d92eaae1?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 9,
        'type': 'track',
        'uri': 'spotify:track:6tX2z3DnwUGTBgctckRDYs'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/5R3xH3a2BADMtpDAQ3LUQU'},
          'href': 'https://api.spotify.com/v1/artists/5R3xH3a2BADMtpDAQ3LUQU',
          'id': '5R3xH3a2BADMtpDAQ3LUQU',
          'name': 'Fantasia',
          'type': 'artist',
          'uri': 'spotify:artist:5R3xH3a2BADMtpDAQ3LUQU'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 252560,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/1oMsZBj6wwKRbWKLjY9HwL'},
        'href': 'https://api.spotify.com/v1/tracks/1oMsZBj6wwKRbWKLjY9HwL',
        'id': '1oMsZBj6wwKRbWKLjY9HwL',
        'is_local': False,
        'name': '4 My Man (feat. Fantasia)',
        'preview_url': 'https://p.scdn.co/mp3-preview/ee85b57aa98da1d7bb7a04d0f522526f8f0b1a2e?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 10,
        'type': 'track',
        'uri': 'spotify:track:1oMsZBj6wwKRbWKLjY9HwL'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 229066,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/32UJmZRXTufLI1X9r28pix'},
        'href': 'https://api.spotify.com/v1/tracks/32UJmZRXTufLI1X9r28pix',
        'id': '32UJmZRXTufLI1X9r28pix',
        'is_local': False,
        'name': "Can't Stop",
        'preview_url': 'https://p.scdn.co/mp3-preview/ef49743eab8d141aee56065f0c160f58ef3f3151?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 11,
        'type': 'track',
        'uri': 'spotify:track:32UJmZRXTufLI1X9r28pix'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 210959,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/4HUGbUuK6lUqI3aGVm4JoP'},
        'href': 'https://api.spotify.com/v1/tracks/4HUGbUuK6lUqI3aGVm4JoP',
        'id': '4HUGbUuK6lUqI3aGVm4JoP',
        'is_local': False,
        'name': 'Teary Eyed',
        'preview_url': 'https://p.scdn.co/mp3-preview/6c7c13d973d13699d662e04ec2443ce19a87d5b8?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 12,
        'type': 'track',
        'uri': 'spotify:track:4HUGbUuK6lUqI3aGVm4JoP'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 188213,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/0Z8taEEMbqDMV0eNmD1ypH'},
        'href': 'https://api.spotify.com/v1/tracks/0Z8taEEMbqDMV0eNmD1ypH',
        'id': '0Z8taEEMbqDMV0eNmD1ypH',
        'is_local': False,
        'name': 'Mommy',
        'preview_url': 'https://p.scdn.co/mp3-preview/c68c38577078df584b7bc16631600bb1984ff539?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 13,
        'type': 'track',
        'uri': 'spotify:track:0Z8taEEMbqDMV0eNmD1ypH'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 174653,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/6XcO3qAAFG9e7DzbgVOEoV'},
        'href': 'https://api.spotify.com/v1/tracks/6XcO3qAAFG9e7DzbgVOEoV',
        'id': '6XcO3qAAFG9e7DzbgVOEoV',
        'is_local': False,
        'name': 'Click Clack',
        'preview_url': 'https://p.scdn.co/mp3-preview/f9d55654ef3f9a8971bb19ddcca92f41a08dac13?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 14,
        'type': 'track',
        'uri': 'spotify:track:6XcO3qAAFG9e7DzbgVOEoV'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 229693,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/2VDLdk7DGQs0pyWadAMSWo'},
        'href': 'https://api.spotify.com/v1/tracks/2VDLdk7DGQs0pyWadAMSWo',
        'id': '2VDLdk7DGQs0pyWadAMSWo',
        'is_local': False,
        'name': 'Time And Time Again',
        'preview_url': 'https://p.scdn.co/mp3-preview/9592b00c62e802ef03c9516309946d3b0f4142af?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 15,
        'type': 'track',
        'uri': 'spotify:track:2VDLdk7DGQs0pyWadAMSWo'},
       {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/0GsZLtEGLQ83TChF3uVhu8'},
          'href': 'https://api.spotify.com/v1/artists/0GsZLtEGLQ83TChF3uVhu8',
          'id': '0GsZLtEGLQ83TChF3uVhu8',
          'name': 'Vybez Cartel',
          'type': 'artist',
          'uri': 'spotify:artist:0GsZLtEGLQ83TChF3uVhu8'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/451jqMEf2cDjJKgw1uBFGe'},
          'href': 'https://api.spotify.com/v1/artists/451jqMEf2cDjJKgw1uBFGe',
          'id': '451jqMEf2cDjJKgw1uBFGe',
          'name': 'M.I.A.',
          'type': 'artist',
          'uri': 'spotify:artist:451jqMEf2cDjJKgw1uBFGe'}],
        'available_markets': ['AD',
         'AE',
         'AR',
         'AT',
         'AU',
         'BE',
         'BG',
         'BH',
         'BO',
         'BR',
         'CA',
         'CH',
         'CL',
         'CO',
         'CR',
         'CY',
         'CZ',
         'DE',
         'DK',
         'DO',
         'DZ',
         'EC',
         'EE',
         'EG',
         'ES',
         'FI',
         'FR',
         'GB',
         'GR',
         'GT',
         'HK',
         'HN',
         'HU',
         'ID',
         'IE',
         'IL',
         'IS',
         'IT',
         'JO',
         'JP',
         'KW',
         'LB',
         'LI',
         'LT',
         'LU',
         'LV',
         'MA',
         'MC',
         'MT',
         'MX',
         'MY',
         'NI',
         'NL',
         'NO',
         'NZ',
         'OM',
         'PA',
         'PE',
         'PH',
         'PL',
         'PS',
         'PT',
         'PY',
         'QA',
         'RO',
         'SA',
         'SE',
         'SG',
         'SK',
         'SV',
         'TH',
         'TN',
         'TR',
         'TW',
         'US',
         'UY',
         'VN',
         'ZA'],
        'disc_number': 1,
        'duration_ms': 312333,
        'explicit': True,
        'external_urls': {'spotify': 'https://open.spotify.com/track/2rKXzis3tBuNQQojmCldkv'},
        'href': 'https://api.spotify.com/v1/tracks/2rKXzis3tBuNQQojmCldkv',
        'id': '2rKXzis3tBuNQQojmCldkv',
        'is_local': False,
        'name': 'Bad Man (feat. Vybez Cartel & M.I.A.)',
        'preview_url': 'https://p.scdn.co/mp3-preview/4d4b82fe55e31fdca17022687fcc97e833e4abea?cid=e6ff82a6418a4191a5b3a95622faf5dd',
        'track_number': 16,
        'type': 'track',
        'uri': 'spotify:track:2rKXzis3tBuNQQojmCldkv'}],
      'limit': 50,
      'next': None,
      'offset': 0,
      'previous': None,
      'total': 16},
     'type': 'album',
     'uri': 'spotify:album:6vV5UrXcfyQD1wu4Qo2I9K'}



- `spotipy.Spotify().audio_features(track_uri)`: Get audio feature information for a single track identified by its unique Spotify ID. (https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)
    - `danceability` (float): Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
    <img src="https://developer.spotify.com/assets/audio/danceability.png" width="600" height="250">
    - `energy` (float): Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
    <img src="https://developer.spotify.com/assets/audio/energy.png" width="600" height="250">
    - `Key` (int): The estimated overall key of the track. Integers map to pitches using standard `Pitch Class notation`. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
    - `loudness` (float): The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
    <img src="https://developer.spotify.com/assets/audio/loudness.png" width="600" height="250">
    - `mode` (int): Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
    - `speechiness` (float): Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
    <img src="https://developer.spotify.com/assets/audio/speechiness.png" width="600" height="250">
    - `acousticness` (float): A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
    <img src="https://developer.spotify.com/assets/audio/acousticness.png" width="600" height="250">
    - `instrumentalness` (float): Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
    <img src="https://developer.spotify.com/assets/audio/instrumentalness.png" width="600" height="250">
    - `liveness` (float): Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
    <img src="https://developer.spotify.com/assets/audio/liveness.png" width="600" height="250">
    - `valence` (float): A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
    <img src="https://developer.spotify.com/assets/audio/valence.png" width="600" height="250">
    - `tempo` (float): The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
    <img src="https://developer.spotify.com/assets/audio/tempo.png" width="600" height="250">
    - `type` (string): The object type, `audio_features`.
    - `id` (string): The `Spotify ID` for the track.
    - `uri` (string): The `Spotify URI` for the track.
    - `track_href` (string): A link to the Web API endpoint providing full details of the track.
    - `analysis_url` (string): An HTTP URL to access the full audio analysis of this track. An access token is required to access this data.
    - `duration_ms` (int): The duration of the track in milliseconds.
    - `time_signature` (int): An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).


- Below shows an example data provided by `spotipy.Spotify().audio_features(track_uri)` for the 0th track in the 0th playlist in the 0th file:



```python
TRACK_URI = playlist_data['playlists'][0]['tracks'][0]['track_uri']
sp.audio_features(TRACK_URI)
```





    [{'danceability': 0.904,
      'energy': 0.813,
      'key': 4,
      'loudness': -7.105,
      'mode': 0,
      'speechiness': 0.121,
      'acousticness': 0.0311,
      'instrumentalness': 0.00697,
      'liveness': 0.0471,
      'valence': 0.81,
      'tempo': 125.461,
      'type': 'audio_features',
      'id': '0UaMYEvWZi0ZqiDOoHU3YI',
      'uri': 'spotify:track:0UaMYEvWZi0ZqiDOoHU3YI',
      'track_href': 'https://api.spotify.com/v1/tracks/0UaMYEvWZi0ZqiDOoHU3YI',
      'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0UaMYEvWZi0ZqiDOoHU3YI',
      'duration_ms': 226864,
      'time_signature': 4}]



- `spotipy.Spotify().artist_related_artists(artist_uri)`: Get Spotify catalog information about artists similar to a given artist. Similarity is based on analysis of the Spotify community's listening history.
(https://developer.spotify.com/documentation/web-api/reference/artists/get-related-artists/)
    - `artists` (array of `artist objects`): Up to 20 related artists to the artist.


- Below shows an example data provided by `spotipy.Spotify().artist_related_artists(artist_uri)` for the artist of the 0th track in the 0th playlist in the 0th file:



```python
ARTIST_URI = playlist_data['playlists'][0]['tracks'][0]['artist_uri']
sp.artist_related_artists(ARTIST_URI)
```





    {'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/5tth2a3v0sWwV1C7bApBdX'},
       'followers': {'href': None, 'total': 516154},
       'genres': ['dance pop',
        'east coast hip hop',
        'escape room',
        'hip hop',
        'hip pop',
        'pop',
        'pop rap',
        'r&b',
        'rap',
        'southern hip hop',
        'trap queen',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/5tth2a3v0sWwV1C7bApBdX',
       'id': '5tth2a3v0sWwV1C7bApBdX',
       'images': [{'height': 1000,
         'url': 'https://i.scdn.co/image/32f4515ff2426fe1e2256cadef56783bfaeb52bd',
         'width': 1000},
        {'height': 640,
         'url': 'https://i.scdn.co/image/40aa4f887d48a0084bbfe2bb7a15f12d1cff5c13',
         'width': 640},
        {'height': 200,
         'url': 'https://i.scdn.co/image/64eed06bbf737e87addaacdaf3a4a0dc97af698c',
         'width': 200},
        {'height': 64,
         'url': 'https://i.scdn.co/image/bdaf058a1e6cfaa3f38911aa02d8ef936702ffee',
         'width': 64}],
       'name': "Lil' Kim",
       'popularity': 69,
       'type': 'artist',
       'uri': 'spotify:artist:5tth2a3v0sWwV1C7bApBdX'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/0urTpYCsixqZwgNTkPJOJ4'},
       'followers': {'href': None, 'total': 1448189},
       'genres': ['dance pop',
        'hip hop',
        'hip pop',
        'indie r&b',
        'neo soul',
        'pop',
        'r&b',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/0urTpYCsixqZwgNTkPJOJ4',
       'id': '0urTpYCsixqZwgNTkPJOJ4',
       'images': [{'height': 1236,
         'url': 'https://i.scdn.co/image/341693fbd0b8ed0e8ae042713598e2c9ea4870dc',
         'width': 1000},
        {'height': 791,
         'url': 'https://i.scdn.co/image/1700f9e23fef0c6601f70a3f972d763fa092653f',
         'width': 640},
        {'height': 247,
         'url': 'https://i.scdn.co/image/bd236cddd586414ed7a95b89241f5b9ad207b122',
         'width': 200},
        {'height': 79,
         'url': 'https://i.scdn.co/image/a1c96f8eec7b192f3a5d6de8ffbda70f5a6e0424',
         'width': 64}],
       'name': 'Aaliyah',
       'popularity': 62,
       'type': 'artist',
       'uri': 'spotify:artist:0urTpYCsixqZwgNTkPJOJ4'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/4d3yvTptO48nOYTPBcPFZC'},
       'followers': {'href': None, 'total': 440646},
       'genres': ['dance pop',
        'hip hop',
        'hip pop',
        'pop rap',
        'r&b',
        'rap',
        'southern hip hop',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/4d3yvTptO48nOYTPBcPFZC',
       'id': '4d3yvTptO48nOYTPBcPFZC',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/68bd11923693d8e6d37f09fd69025b4f5f781ed3',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/40607cee5edc8cf97f3274bf1d539a7169ee534b',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/34bb29ea819b3fca3d551aa6da8e231b4e36acb0',
         'width': 160}],
       'name': 'Eve',
       'popularity': 66,
       'type': 'artist',
       'uri': 'spotify:artist:4d3yvTptO48nOYTPBcPFZC'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/1wvlC6NwleHt1iRD6d5X2C'},
       'followers': {'href': None, 'total': 210954},
       'genres': ['east coast hip hop',
        'hardcore hip hop',
        'hip hop',
        'hip pop',
        'r&b',
        'rap',
        'southern hip hop'],
       'href': 'https://api.spotify.com/v1/artists/1wvlC6NwleHt1iRD6d5X2C',
       'id': '1wvlC6NwleHt1iRD6d5X2C',
       'images': [{'height': 1133,
         'url': 'https://i.scdn.co/image/7b45e618a057dfcad3202190e165956f52eb8004',
         'width': 1000},
        {'height': 725,
         'url': 'https://i.scdn.co/image/ae4905a8c03a0643faf42310aa4962b5374aa206',
         'width': 640},
        {'height': 227,
         'url': 'https://i.scdn.co/image/aeea54fb8403f01965439f5327244bb63c9aed5d',
         'width': 200},
        {'height': 72,
         'url': 'https://i.scdn.co/image/6fb90f1d7b941243a4f3837b952a30165f3767b8',
         'width': 64}],
       'name': 'Foxy Brown',
       'popularity': 61,
       'type': 'artist',
       'uri': 'spotify:artist:1wvlC6NwleHt1iRD6d5X2C'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/6lHL3ubAMgSasKjNqKb8HF'},
       'followers': {'href': None, 'total': 509772},
       'genres': ['dance pop',
        'deep pop r&b',
        'hip pop',
        'neo soul',
        'r&b',
        'southern hip hop',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/6lHL3ubAMgSasKjNqKb8HF',
       'id': '6lHL3ubAMgSasKjNqKb8HF',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/a66bcf14cb2a5367e5f277ba6c9cf4694b958158',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/cf3027f4d8b7a5ec69ff35e9a78f07192682d847',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/1fec3ffe56bd3bede41b0ea763cec9c4b5440a6f',
         'width': 160}],
       'name': 'Mýa',
       'popularity': 65,
       'type': 'artist',
       'uri': 'spotify:artist:6lHL3ubAMgSasKjNqKb8HF'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/05oH07COxkXKIMt6mIPRee'},
       'followers': {'href': None, 'total': 1090512},
       'genres': ['dance pop',
        'deep pop r&b',
        'hip hop',
        'hip pop',
        'indie r&b',
        'neo soul',
        'new jack swing',
        'pop',
        'pop rap',
        'r&b',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/05oH07COxkXKIMt6mIPRee',
       'id': '05oH07COxkXKIMt6mIPRee',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/880d6e3e6fbfbaa065a4e95339686c70bfa09dd5',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/b3432276aa59f43c519d57cd36b7dd955b1fd5aa',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/fe4c49a5f28b000e5b86e8b34c67e5dbd5d98241',
         'width': 160}],
       'name': 'Brandy',
       'popularity': 67,
       'type': 'artist',
       'uri': 'spotify:artist:05oH07COxkXKIMt6mIPRee'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/0IF46mUS8NXjgHabxk2MCM'},
       'followers': {'href': None, 'total': 243477},
       'genres': ['dance pop',
        'electropop',
        'hip pop',
        'indie r&b',
        'neo soul',
        'pop rap',
        'r&b',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/0IF46mUS8NXjgHabxk2MCM',
       'id': '0IF46mUS8NXjgHabxk2MCM',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/977668064535a289d35377405b81673d22355411',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/5ff63728fbd782130dffe4a914ad7aab959362c4',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/cea226f978489b11c4c17105253bf92c31444715',
         'width': 160}],
       'name': 'Kelis',
       'popularity': 65,
       'type': 'artist',
       'uri': 'spotify:artist:0IF46mUS8NXjgHabxk2MCM'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/08rMCq2ek1YjdDBsCPVH2s'},
       'followers': {'href': None, 'total': 360947},
       'genres': ['dance pop',
        'deep pop r&b',
        'hip hop',
        'hip pop',
        'indie r&b',
        'neo soul',
        'r&b',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/08rMCq2ek1YjdDBsCPVH2s',
       'id': '08rMCq2ek1YjdDBsCPVH2s',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/91c9bea2e8b07d3af8e3696fcb45e54febeafab6',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/0696b76c69d2dfa1d9c97e0aec0e226b9e59b15e',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/2dd03479fc64ddfb22efe93bd2b03d4e8b59851b',
         'width': 160}],
       'name': 'Amerie',
       'popularity': 56,
       'type': 'artist',
       'uri': 'spotify:artist:08rMCq2ek1YjdDBsCPVH2s'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/0TImkz4nPqjegtVSMZnMRq'},
       'followers': {'href': None, 'total': 1353869},
       'genres': ['atl hip hop',
        'dance pop',
        'girl group',
        'hip pop',
        'maskandi',
        'new jack swing',
        'pop',
        'r&b',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/0TImkz4nPqjegtVSMZnMRq',
       'id': '0TImkz4nPqjegtVSMZnMRq',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/0a92641cc8acb39e4aee7e959cb2a42287ea4801',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/0f9732ac88b697810043c330cb8f01fcf0a41ac0',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/fa845368b2015053f59ce3eda1550601a01c6dce',
         'width': 160}],
       'name': 'TLC',
       'popularity': 72,
       'type': 'artist',
       'uri': 'spotify:artist:0TImkz4nPqjegtVSMZnMRq'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/4PrinKSrmILmo0kERG0Ogn'},
       'followers': {'href': None, 'total': 354384},
       'genres': ['deep pop r&b',
        'dirty south rap',
        'hip pop',
        'miami hip hop',
        'pop rap',
        'r&b',
        'rap',
        'southern hip hop',
        'trap music',
        'trap queen'],
       'href': 'https://api.spotify.com/v1/artists/4PrinKSrmILmo0kERG0Ogn',
       'id': '4PrinKSrmILmo0kERG0Ogn',
       'images': [{'height': 1500,
         'url': 'https://i.scdn.co/image/68ac67ac6c6b4c1c575ac5388db7c1e290b36f03',
         'width': 1000},
        {'height': 960,
         'url': 'https://i.scdn.co/image/c2599f3a0656caf6d6ced2177ae1532f78cbc803',
         'width': 640},
        {'height': 300,
         'url': 'https://i.scdn.co/image/d7e9a4412073e06461969e851ef921a319feab50',
         'width': 200},
        {'height': 96,
         'url': 'https://i.scdn.co/image/4b819c35c65fa107c60a38edd050a2ab86780f84',
         'width': 64}],
       'name': 'Trina',
       'popularity': 57,
       'type': 'artist',
       'uri': 'spotify:artist:4PrinKSrmILmo0kERG0Ogn'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/2NdeV5rLm47xAvogXrYhJX'},
       'followers': {'href': None, 'total': 3481027},
       'genres': ['dance pop',
        'hip hop',
        'hip pop',
        'pop',
        'pop rap',
        'post-teen pop',
        'r&b',
        'rap',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/2NdeV5rLm47xAvogXrYhJX',
       'id': '2NdeV5rLm47xAvogXrYhJX',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/26da3944201bfadf0095bd6b469bb8e9f53b81e1',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/bfa9c7251cdd0518f370d0607a164fd76b878a28',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/4a1b8b28418902a57d09607b010c0ff40ff71288',
         'width': 160}],
       'name': 'Ciara',
       'popularity': 75,
       'type': 'artist',
       'uri': 'spotify:artist:2NdeV5rLm47xAvogXrYhJX'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/1urjDGTd4iBze91Z1W1gu7'},
       'followers': {'href': None, 'total': 199015},
       'genres': ['hip hop',
        'hip pop',
        'neo soul',
        'new jack swing',
        'r&b',
        'southern hip hop',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/1urjDGTd4iBze91Z1W1gu7',
       'id': '1urjDGTd4iBze91Z1W1gu7',
       'images': [{'height': 1059,
         'url': 'https://i.scdn.co/image/91e596f44efa630bb9489dc57c2aa571e2a3b095',
         'width': 1000},
        {'height': 678,
         'url': 'https://i.scdn.co/image/559f3c54be2d1b7265f00d6fd0dc6e5d3e30e903',
         'width': 640},
        {'height': 212,
         'url': 'https://i.scdn.co/image/fe03fcb3f7b967647e0063a3f5d2a4e265da65b8',
         'width': 200},
        {'height': 68,
         'url': 'https://i.scdn.co/image/c82a76c8b0c2b8fbbb14cc3833076265cf36ea67',
         'width': 64}],
       'name': 'Total',
       'popularity': 59,
       'type': 'artist',
       'uri': 'spotify:artist:1urjDGTd4iBze91Z1W1gu7'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/2I1bnmb9VQEQGKHxvr0gSf'},
       'followers': {'href': None, 'total': 153994},
       'genres': ['hip hop',
        'hip pop',
        'new jack swing',
        'r&b',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/2I1bnmb9VQEQGKHxvr0gSf',
       'id': '2I1bnmb9VQEQGKHxvr0gSf',
       'images': [{'height': 1228,
         'url': 'https://i.scdn.co/image/588abbff92446ec0bd740cf837a83c9494d5dfec',
         'width': 1000},
        {'height': 786,
         'url': 'https://i.scdn.co/image/2eb2e0c8a8a8e81e658ff26fa812518ec24b6077',
         'width': 640},
        {'height': 246,
         'url': 'https://i.scdn.co/image/a74ccf5ed61a19bf61f4366ba260c69d697c0a2b',
         'width': 200},
        {'height': 79,
         'url': 'https://i.scdn.co/image/8fb6449a7b9a17651756dd3e4fd4209aff06be10',
         'width': 64}],
       'name': 'Da Brat',
       'popularity': 53,
       'type': 'artist',
       'uri': 'spotify:artist:2I1bnmb9VQEQGKHxvr0gSf'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/4qwGe91Bz9K2T8jXTZ815W'},
       'followers': {'href': None, 'total': 1281120},
       'genres': ['dance pop',
        'hip pop',
        'motown',
        'neo soul',
        'pop',
        'r&b',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/4qwGe91Bz9K2T8jXTZ815W',
       'id': '4qwGe91Bz9K2T8jXTZ815W',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/fb98451f3e31de67c35bcb22f0551728b180c3d4',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/3de280e4c4e78b21e10509bd6973e3ea16406e20',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/cc361cfe4c212e391f7aa0eb8864e25b95a4029c',
         'width': 160}],
       'name': 'Janet Jackson',
       'popularity': 70,
       'type': 'artist',
       'uri': 'spotify:artist:4qwGe91Bz9K2T8jXTZ815W'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/6nzxy2wXs6tLgzEtqOkEi2'},
       'followers': {'href': None, 'total': 978292},
       'genres': ['dance pop',
        'deep pop r&b',
        'hip pop',
        'indie r&b',
        'neo soul',
        'new jack swing',
        'pop rap',
        'quiet storm',
        'r&b',
        'rap',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/6nzxy2wXs6tLgzEtqOkEi2',
       'id': '6nzxy2wXs6tLgzEtqOkEi2',
       'images': [{'height': 720,
         'url': 'https://i.scdn.co/image/328896170b380ff5fc06795a857c175e68bb526c',
         'width': 706},
        {'height': 653,
         'url': 'https://i.scdn.co/image/af8772ceeab75d0cbfea929119b2a01626bd4b9d',
         'width': 640},
        {'height': 204,
         'url': 'https://i.scdn.co/image/7a58c4aba7443a5fb420274f1e7521eb2692692f',
         'width': 200},
        {'height': 65,
         'url': 'https://i.scdn.co/image/e8d4b95e7b561059667717919d79c0ae96d5205a',
         'width': 64}],
       'name': 'Monica',
       'popularity': 64,
       'type': 'artist',
       'uri': 'spotify:artist:6nzxy2wXs6tLgzEtqOkEi2'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/6zDBeei6hHRiZdAJ6zoTCo'},
       'followers': {'href': None, 'total': 240918},
       'genres': ['hip pop', 'indie r&b', 'neo soul', 'r&b', 'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/6zDBeei6hHRiZdAJ6zoTCo',
       'id': '6zDBeei6hHRiZdAJ6zoTCo',
       'images': [{'height': 1272,
         'url': 'https://i.scdn.co/image/9cf78b831ef3a4c193148c73cafaf847010467ab',
         'width': 1000},
        {'height': 814,
         'url': 'https://i.scdn.co/image/76d687ea0fab91919f2639989fbb597aa83f26cb',
         'width': 640},
        {'height': 254,
         'url': 'https://i.scdn.co/image/8521bce8b1ff6972d4a4c4b9afbefc272aad3add',
         'width': 200},
        {'height': 81,
         'url': 'https://i.scdn.co/image/6566a016dab54dfd24edba56939acb911054be08',
         'width': 64}],
       'name': 'Tweet',
       'popularity': 50,
       'type': 'artist',
       'uri': 'spotify:artist:6zDBeei6hHRiZdAJ6zoTCo'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/1YfEcTuGvBQ8xSD1f53UnK'},
       'followers': {'href': None, 'total': 1002041},
       'genres': ['dirty south rap',
        'east coast hip hop',
        'hardcore hip hop',
        'hip hop',
        'hip pop',
        'pop rap',
        'rap'],
       'href': 'https://api.spotify.com/v1/artists/1YfEcTuGvBQ8xSD1f53UnK',
       'id': '1YfEcTuGvBQ8xSD1f53UnK',
       'images': [{'height': 730,
         'url': 'https://i.scdn.co/image/d4c28fb970ac89443e6ce401222de7ad30382ee5',
         'width': 1000},
        {'height': 467,
         'url': 'https://i.scdn.co/image/4c1251b4e737a29def84778a2cc627daf3bfd42c',
         'width': 640},
        {'height': 146,
         'url': 'https://i.scdn.co/image/7e2375b5ae64bcd0792e44303f1c67e3f98ebeae',
         'width': 200},
        {'height': 47,
         'url': 'https://i.scdn.co/image/7e4572c7979e272293fa0490915b472b74235e94',
         'width': 64}],
       'name': 'Busta Rhymes',
       'popularity': 75,
       'type': 'artist',
       'uri': 'spotify:artist:1YfEcTuGvBQ8xSD1f53UnK'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/7wqtxqI3eo7Gn1P7SpP6cQ'},
       'followers': {'href': None, 'total': 298093},
       'genres': ['dance pop',
        'girl group',
        'hip hop',
        'hip house',
        'hip pop',
        'old school hip hop',
        'pop rap',
        'rap'],
       'href': 'https://api.spotify.com/v1/artists/7wqtxqI3eo7Gn1P7SpP6cQ',
       'id': '7wqtxqI3eo7Gn1P7SpP6cQ',
       'images': [{'height': 1218,
         'url': 'https://i.scdn.co/image/6ff07d51815e67b11f8c2e4a4bebc2b6b3bbe7eb',
         'width': 1000},
        {'height': 779,
         'url': 'https://i.scdn.co/image/8a4ea0b462e4ec2ba81108f5a4e647062535c363',
         'width': 640},
        {'height': 244,
         'url': 'https://i.scdn.co/image/53a0bfc38d12455f9cf065df9b5126dc81ba92ac',
         'width': 200},
        {'height': 78,
         'url': 'https://i.scdn.co/image/a47af5d6cbabbb22b4819788087bd04a8435b249',
         'width': 64}],
       'name': 'Salt-N-Pepa',
       'popularity': 63,
       'type': 'artist',
       'uri': 'spotify:artist:7wqtxqI3eo7Gn1P7SpP6cQ'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/7r8RF1tN2A4CiGEplkp1oP'},
       'followers': {'href': None, 'total': 1120256},
       'genres': ['dance pop',
        'hip hop',
        'hip pop',
        'neo soul',
        'new jack swing',
        'pop rap',
        'r&b',
        'rap',
        'southern hip hop',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/7r8RF1tN2A4CiGEplkp1oP',
       'id': '7r8RF1tN2A4CiGEplkp1oP',
       'images': [{'height': 931,
         'url': 'https://i.scdn.co/image/c8bc9979554744c06b3c62b7e94b82cd9303ec81',
         'width': 600},
        {'height': 310,
         'url': 'https://i.scdn.co/image/9617d45aa1bd6725f9418cc54ff008ddbfedede3',
         'width': 200},
        {'height': 99,
         'url': 'https://i.scdn.co/image/92c562bd9618c0d00872cfc163f0b4d44fbe5259',
         'width': 64}],
       'name': 'Ginuwine',
       'popularity': 69,
       'type': 'artist',
       'uri': 'spotify:artist:7r8RF1tN2A4CiGEplkp1oP'},
      {'external_urls': {'spotify': 'https://open.spotify.com/artist/5fikk4h5qbEebqK2Fc6e48'},
       'followers': {'href': None, 'total': 432630},
       'genres': ['dance pop',
        'girl group',
        'hip pop',
        'neo soul',
        'new jack swing',
        'quiet storm',
        'r&b',
        'urban contemporary'],
       'href': 'https://api.spotify.com/v1/artists/5fikk4h5qbEebqK2Fc6e48',
       'id': '5fikk4h5qbEebqK2Fc6e48',
       'images': [{'height': 640,
         'url': 'https://i.scdn.co/image/72fc22ee4af38947cb314d3b7e0c1a6be10fc504',
         'width': 640},
        {'height': 320,
         'url': 'https://i.scdn.co/image/4c484ce852a2d386eae9af0af50214b4cbf30dcc',
         'width': 320},
        {'height': 160,
         'url': 'https://i.scdn.co/image/e8e4e601e701c768047a82ac83c51b4aa8019bfc',
         'width': 160}],
       'name': 'En Vogue',
       'popularity': 60,
       'type': 'artist',
       'uri': 'spotify:artist:5fikk4h5qbEebqK2Fc6e48'}]}



- `spotipy.Spotify().artist_top_tracks(artist_uri)`: Get Spotify catalog information about an artist’s top tracks by country.
(https://developer.spotify.com/documentation/web-api/reference/artists/get-artists-top-tracks)
    - `tracks` (array of `track objects`): Up to 10 top tracks of the artist.


- Below shows an example data provided by `spotipy.Spotify().artist_top_tracks(artist_uri)` for the artist of the 0th track in the 0th playlist in the 0th file:



```python
ARTIST_URI = playlist_data['playlists'][0]['tracks'][0]['artist_uri']
sp.artist_top_tracks(ARTIST_URI)
```





    {'tracks': [{'album': {'album_type': 'album',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/20t54K6C80QQH7vbcpfJcP'},
        'href': 'https://api.spotify.com/v1/albums/20t54K6C80QQH7vbcpfJcP',
        'id': '20t54K6C80QQH7vbcpfJcP',
        'images': [{'height': 633,
          'url': 'https://i.scdn.co/image/c94d87f743f11254bdd0ffc5434b59dba851d7d6',
          'width': 640},
         {'height': 297,
          'url': 'https://i.scdn.co/image/0507be05842344647ef44aedd1ccd1f0eaa1e17f',
          'width': 300},
         {'height': 63,
          'url': 'https://i.scdn.co/image/462dd41dd28fd3d7a4e74c24ef542c7e5db2f384',
          'width': 64}],
        'name': 'Miss E...So Addictive',
        'release_date': '2001-05-14',
        'release_date_precision': 'day',
        'total_tracks': 18,
        'type': 'album',
        'uri': 'spotify:album:20t54K6C80QQH7vbcpfJcP'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
       'disc_number': 1,
       'duration_ms': 211120,
       'explicit': True,
       'external_ids': {'isrc': 'USEW20100036'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/6zsk6uF3MxfIeHPlubKBvR'},
       'href': 'https://api.spotify.com/v1/tracks/6zsk6uF3MxfIeHPlubKBvR',
       'id': '6zsk6uF3MxfIeHPlubKBvR',
       'is_local': False,
       'is_playable': True,
       'name': 'Get Ur Freak On',
       'popularity': 70,
       'preview_url': 'https://p.scdn.co/mp3-preview/377a872a710350a8aa969597d4a5a9979fa37d54?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 5,
       'type': 'track',
       'uri': 'spotify:track:6zsk6uF3MxfIeHPlubKBvR'},
      {'album': {'album_type': 'album',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/6DeU398qrJ1bLuryetSmup'},
        'href': 'https://api.spotify.com/v1/albums/6DeU398qrJ1bLuryetSmup',
        'id': '6DeU398qrJ1bLuryetSmup',
        'images': [{'height': 630,
          'url': 'https://i.scdn.co/image/6216880447ea80173eed7ec36ed0b0745e33d955',
          'width': 640},
         {'height': 295,
          'url': 'https://i.scdn.co/image/2e34cd28dd34b1ec0eb6af5d1d1debe632f48fc3',
          'width': 300},
         {'height': 63,
          'url': 'https://i.scdn.co/image/1940ef293906e806e2e48cc0d21081685636325f',
          'width': 64}],
        'name': 'Under Construction',
        'release_date': '2002-11-11',
        'release_date_precision': 'day',
        'total_tracks': 14,
        'type': 'album',
        'uri': 'spotify:album:6DeU398qrJ1bLuryetSmup'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
       'disc_number': 1,
       'duration_ms': 263226,
       'explicit': True,
       'external_ids': {'isrc': 'USEE10240730'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/3jagJCUbdqhDSPuxP8cAqF'},
       'href': 'https://api.spotify.com/v1/tracks/3jagJCUbdqhDSPuxP8cAqF',
       'id': '3jagJCUbdqhDSPuxP8cAqF',
       'is_local': False,
       'is_playable': True,
       'name': 'Work It',
       'popularity': 68,
       'preview_url': 'https://p.scdn.co/mp3-preview/a007b4cabe8441d6d878da45bc3b71471848101e?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 4,
       'type': 'track',
       'uri': 'spotify:track:3jagJCUbdqhDSPuxP8cAqF'},
      {'album': {'album_type': 'single',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/7HV2RI2qNug4EcQqLbCAKS'},
          'href': 'https://api.spotify.com/v1/artists/7HV2RI2qNug4EcQqLbCAKS',
          'id': '7HV2RI2qNug4EcQqLbCAKS',
          'name': 'Keala Settle',
          'type': 'artist',
          'uri': 'spotify:artist:7HV2RI2qNug4EcQqLbCAKS'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/6LqNN22kT3074XbTVUrhzX'},
          'href': 'https://api.spotify.com/v1/artists/6LqNN22kT3074XbTVUrhzX',
          'id': '6LqNN22kT3074XbTVUrhzX',
          'name': 'Kesha',
          'type': 'artist',
          'uri': 'spotify:artist:6LqNN22kT3074XbTVUrhzX'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/2Ed0G2CXxbwS1W3OEAhmBO'},
        'href': 'https://api.spotify.com/v1/albums/2Ed0G2CXxbwS1W3OEAhmBO',
        'id': '2Ed0G2CXxbwS1W3OEAhmBO',
        'images': [{'height': 640,
          'url': 'https://i.scdn.co/image/d6b9f37838bb6f4eb8ffca96fe18b20b3d23f2f8',
          'width': 640},
         {'height': 300,
          'url': 'https://i.scdn.co/image/fe59f47047ff29fc36ad2ee01a50b657cc206c94',
          'width': 300},
         {'height': 64,
          'url': 'https://i.scdn.co/image/e2b730b0a50740808bf6ac5e1a4cb67670bf096a',
          'width': 64}],
        'name': 'This Is Me (The Reimagined Remix) [with Keala Settle, Kesha & Missy Elliott]',
        'release_date': '2018-11-16',
        'release_date_precision': 'day',
        'total_tracks': 1,
        'type': 'album',
        'uri': 'spotify:album:2Ed0G2CXxbwS1W3OEAhmBO'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/7HV2RI2qNug4EcQqLbCAKS'},
         'href': 'https://api.spotify.com/v1/artists/7HV2RI2qNug4EcQqLbCAKS',
         'id': '7HV2RI2qNug4EcQqLbCAKS',
         'name': 'Keala Settle',
         'type': 'artist',
         'uri': 'spotify:artist:7HV2RI2qNug4EcQqLbCAKS'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/6LqNN22kT3074XbTVUrhzX'},
         'href': 'https://api.spotify.com/v1/artists/6LqNN22kT3074XbTVUrhzX',
         'id': '6LqNN22kT3074XbTVUrhzX',
         'name': 'Kesha',
         'type': 'artist',
         'uri': 'spotify:artist:6LqNN22kT3074XbTVUrhzX'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
       'disc_number': 1,
       'duration_ms': 265733,
       'explicit': False,
       'external_ids': {'isrc': 'USAT21811553'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/3WazefEHBjVsgnceB9TuAp'},
       'href': 'https://api.spotify.com/v1/tracks/3WazefEHBjVsgnceB9TuAp',
       'id': '3WazefEHBjVsgnceB9TuAp',
       'is_local': False,
       'is_playable': True,
       'name': 'This Is Me (The Reimagined Remix) [with Keala Settle, Kesha & Missy Elliott]',
       'popularity': 59,
       'preview_url': 'https://p.scdn.co/mp3-preview/0a06ff1bb7508d72f9948c1c6c28eb8bd0ce2460?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 1,
       'type': 'track',
       'uri': 'spotify:track:3WazefEHBjVsgnceB9TuAp'},
      {'album': {'album_type': 'album',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/6vV5UrXcfyQD1wu4Qo2I9K'},
        'href': 'https://api.spotify.com/v1/albums/6vV5UrXcfyQD1wu4Qo2I9K',
        'id': '6vV5UrXcfyQD1wu4Qo2I9K',
        'images': [{'height': 640,
          'url': 'https://i.scdn.co/image/608d68c28beb4d3fb2891ba983965c28d550f592',
          'width': 640},
         {'height': 300,
          'url': 'https://i.scdn.co/image/facb5dc58151f589a8328d0aa148425e67e260ea',
          'width': 300},
         {'height': 64,
          'url': 'https://i.scdn.co/image/03b79b1f4336fe04ec3b62d9fd01f2034bff4fb5',
          'width': 64}],
        'name': 'The Cookbook',
        'release_date': '2005-07-04',
        'release_date_precision': 'day',
        'total_tracks': 16,
        'type': 'album',
        'uri': 'spotify:album:6vV5UrXcfyQD1wu4Qo2I9K'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/2NdeV5rLm47xAvogXrYhJX'},
         'href': 'https://api.spotify.com/v1/artists/2NdeV5rLm47xAvogXrYhJX',
         'id': '2NdeV5rLm47xAvogXrYhJX',
         'name': 'Ciara',
         'type': 'artist',
         'uri': 'spotify:artist:2NdeV5rLm47xAvogXrYhJX'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/15GGbJKqC6w0VYyAJtjej6'},
         'href': 'https://api.spotify.com/v1/artists/15GGbJKqC6w0VYyAJtjej6',
         'id': '15GGbJKqC6w0VYyAJtjej6',
         'name': 'Fatman Scoop',
         'type': 'artist',
         'uri': 'spotify:artist:15GGbJKqC6w0VYyAJtjej6'}],
       'disc_number': 1,
       'duration_ms': 226863,
       'explicit': True,
       'external_ids': {'isrc': 'USEE10414022'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/0UaMYEvWZi0ZqiDOoHU3YI'},
       'href': 'https://api.spotify.com/v1/tracks/0UaMYEvWZi0ZqiDOoHU3YI',
       'id': '0UaMYEvWZi0ZqiDOoHU3YI',
       'is_local': False,
       'is_playable': True,
       'name': 'Lose Control (feat. Ciara & Fat Man Scoop)',
       'popularity': 62,
       'preview_url': 'https://p.scdn.co/mp3-preview/fe79cac4e5f67a23d54ea5e884d86af1b5248a88?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 4,
       'type': 'track',
       'uri': 'spotify:track:0UaMYEvWZi0ZqiDOoHU3YI'},
      {'album': {'album_type': 'album',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/6UkdyvPElK6JDkyeRClbI2'},
        'href': 'https://api.spotify.com/v1/albums/6UkdyvPElK6JDkyeRClbI2',
        'id': '6UkdyvPElK6JDkyeRClbI2',
        'images': [{'height': 640,
          'url': 'https://i.scdn.co/image/a19fe7403b04786737942bde0a21e20e72ed6ef1',
          'width': 640},
         {'height': 300,
          'url': 'https://i.scdn.co/image/1c8500e05d9a28995c3bb1dda0bb831e7c2e3674',
          'width': 300},
         {'height': 64,
          'url': 'https://i.scdn.co/image/14f9d8b9daabb35663ee9f9b8ceb0128c7408189',
          'width': 64}],
        'name': 'Supa Dupa Fly',
        'release_date': '1997-07-11',
        'release_date_precision': 'day',
        'total_tracks': 17,
        'type': 'album',
        'uri': 'spotify:album:6UkdyvPElK6JDkyeRClbI2'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
       'disc_number': 1,
       'duration_ms': 246173,
       'explicit': True,
       'external_ids': {'isrc': 'USEW19706204'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/2WRzpLD8qDRrxMXc63E5WJ'},
       'href': 'https://api.spotify.com/v1/tracks/2WRzpLD8qDRrxMXc63E5WJ',
       'id': '2WRzpLD8qDRrxMXc63E5WJ',
       'is_local': False,
       'is_playable': True,
       'name': 'The Rain (Supa Dupa Fly)',
       'popularity': 57,
       'preview_url': 'https://p.scdn.co/mp3-preview/dd17a9032afd28d262c39df600881108c214fa97?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 4,
       'type': 'track',
       'uri': 'spotify:track:2WRzpLD8qDRrxMXc63E5WJ'},
      {'album': {'album_type': 'single',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2NdeV5rLm47xAvogXrYhJX'},
          'href': 'https://api.spotify.com/v1/artists/2NdeV5rLm47xAvogXrYhJX',
          'id': '2NdeV5rLm47xAvogXrYhJX',
          'name': 'Ciara',
          'type': 'artist',
          'uri': 'spotify:artist:2NdeV5rLm47xAvogXrYhJX'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
         {'external_urls': {'spotify': 'https://open.spotify.com/artist/15GGbJKqC6w0VYyAJtjej6'},
          'href': 'https://api.spotify.com/v1/artists/15GGbJKqC6w0VYyAJtjej6',
          'id': '15GGbJKqC6w0VYyAJtjej6',
          'name': 'Fatman Scoop',
          'type': 'artist',
          'uri': 'spotify:artist:15GGbJKqC6w0VYyAJtjej6'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/13iiYB1fvhrxbybtxNYrxw'},
        'href': 'https://api.spotify.com/v1/albums/13iiYB1fvhrxbybtxNYrxw',
        'id': '13iiYB1fvhrxbybtxNYrxw',
        'images': [{'height': 640,
          'url': 'https://i.scdn.co/image/ba4ab6be0be82c7c9710c476054d5d2d8e2f41c4',
          'width': 640},
         {'height': 300,
          'url': 'https://i.scdn.co/image/052029874c8ccb5f8092b141ce92d2614774909f',
          'width': 300},
         {'height': 64,
          'url': 'https://i.scdn.co/image/6be148f19098f2403d36d28d21df8293670e4ac6',
          'width': 64}],
        'name': 'Level Up (feat. Missy Elliott & Fatman Scoop) [Remix]',
        'release_date': '2018-07-27',
        'release_date_precision': 'day',
        'total_tracks': 1,
        'type': 'album',
        'uri': 'spotify:album:13iiYB1fvhrxbybtxNYrxw'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2NdeV5rLm47xAvogXrYhJX'},
         'href': 'https://api.spotify.com/v1/artists/2NdeV5rLm47xAvogXrYhJX',
         'id': '2NdeV5rLm47xAvogXrYhJX',
         'name': 'Ciara',
         'type': 'artist',
         'uri': 'spotify:artist:2NdeV5rLm47xAvogXrYhJX'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/15GGbJKqC6w0VYyAJtjej6'},
         'href': 'https://api.spotify.com/v1/artists/15GGbJKqC6w0VYyAJtjej6',
         'id': '15GGbJKqC6w0VYyAJtjej6',
         'name': 'Fatman Scoop',
         'type': 'artist',
         'uri': 'spotify:artist:15GGbJKqC6w0VYyAJtjej6'}],
       'disc_number': 1,
       'duration_ms': 229629,
       'explicit': False,
       'external_ids': {'isrc': 'ZZOPM1800421'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/0aBsXZLJDvn0QWfcIqBXq8'},
       'href': 'https://api.spotify.com/v1/tracks/0aBsXZLJDvn0QWfcIqBXq8',
       'id': '0aBsXZLJDvn0QWfcIqBXq8',
       'is_local': False,
       'is_playable': True,
       'name': 'Level Up (feat. Missy Elliott & Fatman Scoop) - Remix',
       'popularity': 57,
       'preview_url': 'https://p.scdn.co/mp3-preview/a8c75c474ccf8f33eb0553d25bedc509c1becb18?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 1,
       'type': 'track',
       'uri': 'spotify:track:0aBsXZLJDvn0QWfcIqBXq8'},
      {'album': {'album_type': 'single',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/5F4BPEd7S8ZNZetT2fiMwO'},
        'href': 'https://api.spotify.com/v1/albums/5F4BPEd7S8ZNZetT2fiMwO',
        'id': '5F4BPEd7S8ZNZetT2fiMwO',
        'images': [{'height': 640,
          'url': 'https://i.scdn.co/image/78529b6cdf2ab396384f9b880af0be4a2142cb39',
          'width': 640},
         {'height': 300,
          'url': 'https://i.scdn.co/image/31b752f6e7ed5762c0e87e80ecdc86ea9ade2f6d',
          'width': 300},
         {'height': 64,
          'url': 'https://i.scdn.co/image/410d44374975ef7e7055c872507156f0ef4fa731',
          'width': 64}],
        'name': 'WTF (Where They From) [feat. Pharrell Williams]',
        'release_date': '2015-11-12',
        'release_date_precision': 'day',
        'total_tracks': 1,
        'type': 'album',
        'uri': 'spotify:album:5F4BPEd7S8ZNZetT2fiMwO'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/2RdwBSPQiwcmiDo9kixcl8'},
         'href': 'https://api.spotify.com/v1/artists/2RdwBSPQiwcmiDo9kixcl8',
         'id': '2RdwBSPQiwcmiDo9kixcl8',
         'name': 'Pharrell Williams',
         'type': 'artist',
         'uri': 'spotify:artist:2RdwBSPQiwcmiDo9kixcl8'}],
       'disc_number': 1,
       'duration_ms': 192772,
       'explicit': True,
       'external_ids': {'isrc': 'USEE11500498'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/7IAa7vUJ11STN7le8XaxsH'},
       'href': 'https://api.spotify.com/v1/tracks/7IAa7vUJ11STN7le8XaxsH',
       'id': '7IAa7vUJ11STN7le8XaxsH',
       'is_local': False,
       'is_playable': True,
       'name': 'WTF (Where They From) [feat. Pharrell Williams]',
       'popularity': 57,
       'preview_url': 'https://p.scdn.co/mp3-preview/9fc9f52fc4e85a8976c7dfa121ea8b1e614d90bc?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 1,
       'type': 'track',
       'uri': 'spotify:track:7IAa7vUJ11STN7le8XaxsH'},
      {'album': {'album_type': 'album',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/4ffXByMAjLpd25ZyzEJNMK'},
        'href': 'https://api.spotify.com/v1/albums/4ffXByMAjLpd25ZyzEJNMK',
        'id': '4ffXByMAjLpd25ZyzEJNMK',
        'images': [{'height': 640,
          'url': 'https://i.scdn.co/image/43d78a4d15cf632c716ce0c2eb2ae2ba6b1eb4a3',
          'width': 640},
         {'height': 300,
          'url': 'https://i.scdn.co/image/3e557fa2a3f6c2adfde1ae88d3a11dce9d1a907a',
          'width': 300},
         {'height': 64,
          'url': 'https://i.scdn.co/image/402d6600213d6efea0b4689e1a6dfce4a57eb1c7',
          'width': 64}],
        'name': 'This Is Not A Test!',
        'release_date': '2003-11-01',
        'release_date_precision': 'day',
        'total_tracks': 16,
        'type': 'album',
        'uri': 'spotify:album:4ffXByMAjLpd25ZyzEJNMK'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
       'disc_number': 1,
       'duration_ms': 217200,
       'explicit': True,
       'external_ids': {'isrc': 'USEE10301720'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/1MaI6NwdrqnE3mRzOYTpoo'},
       'href': 'https://api.spotify.com/v1/tracks/1MaI6NwdrqnE3mRzOYTpoo',
       'id': '1MaI6NwdrqnE3mRzOYTpoo',
       'is_local': False,
       'is_playable': True,
       'name': 'Bomb Intro / Pass That Dutch',
       'popularity': 54,
       'preview_url': 'https://p.scdn.co/mp3-preview/69d15662b2487fe8348e164d9f4eb518fbfb90b7?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 2,
       'type': 'track',
       'uri': 'spotify:track:1MaI6NwdrqnE3mRzOYTpoo'},
      {'album': {'album_type': 'single',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/2gwnDKLNmXaF1LYVxRiRmB'},
        'href': 'https://api.spotify.com/v1/albums/2gwnDKLNmXaF1LYVxRiRmB',
        'id': '2gwnDKLNmXaF1LYVxRiRmB',
        'images': [{'height': 640,
          'url': 'https://i.scdn.co/image/9fbe99dd3330b2b7e855c853ec31f7c6c9a6c6e0',
          'width': 640},
         {'height': 300,
          'url': 'https://i.scdn.co/image/55bd4cc1ac457045f1ce2ab04a3509997fb6ca4a',
          'width': 300},
         {'height': 64,
          'url': 'https://i.scdn.co/image/6afb876749f145f763beaec030099f30273957d5',
          'width': 64}],
        'name': "I'm Better (feat. Lamb)",
        'release_date': '2017-01-27',
        'release_date_precision': 'day',
        'total_tracks': 1,
        'type': 'album',
        'uri': 'spotify:album:2gwnDKLNmXaF1LYVxRiRmB'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/5ExK2Kx7BHM8ABGcdJNW3r'},
         'href': 'https://api.spotify.com/v1/artists/5ExK2Kx7BHM8ABGcdJNW3r',
         'id': '5ExK2Kx7BHM8ABGcdJNW3r',
         'name': 'Lamb',
         'type': 'artist',
         'uri': 'spotify:artist:5ExK2Kx7BHM8ABGcdJNW3r'}],
       'disc_number': 1,
       'duration_ms': 213120,
       'explicit': True,
       'external_ids': {'isrc': 'USEE11700002'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/2Kf9fwIOwZwd6Aw7OxfkF0'},
       'href': 'https://api.spotify.com/v1/tracks/2Kf9fwIOwZwd6Aw7OxfkF0',
       'id': '2Kf9fwIOwZwd6Aw7OxfkF0',
       'is_local': False,
       'is_playable': True,
       'name': "I'm Better (feat. Lamb)",
       'popularity': 53,
       'preview_url': 'https://p.scdn.co/mp3-preview/d097de460062166e4c99725c7ac6e22baa72ee58?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 1,
       'type': 'track',
       'uri': 'spotify:track:2Kf9fwIOwZwd6Aw7OxfkF0'},
      {'album': {'album_type': 'album',
        'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
          'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
          'id': '2wIVse2owClT7go1WT98tk',
          'name': 'Missy Elliott',
          'type': 'artist',
          'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
        'external_urls': {'spotify': 'https://open.spotify.com/album/6DeU398qrJ1bLuryetSmup'},
        'href': 'https://api.spotify.com/v1/albums/6DeU398qrJ1bLuryetSmup',
        'id': '6DeU398qrJ1bLuryetSmup',
        'images': [{'height': 630,
          'url': 'https://i.scdn.co/image/6216880447ea80173eed7ec36ed0b0745e33d955',
          'width': 640},
         {'height': 295,
          'url': 'https://i.scdn.co/image/2e34cd28dd34b1ec0eb6af5d1d1debe632f48fc3',
          'width': 300},
         {'height': 63,
          'url': 'https://i.scdn.co/image/1940ef293906e806e2e48cc0d21081685636325f',
          'width': 64}],
        'name': 'Under Construction',
        'release_date': '2002-11-11',
        'release_date_precision': 'day',
        'total_tracks': 14,
        'type': 'album',
        'uri': 'spotify:album:6DeU398qrJ1bLuryetSmup'},
       'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
         'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
         'id': '2wIVse2owClT7go1WT98tk',
         'name': 'Missy Elliott',
         'type': 'artist',
         'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'},
        {'external_urls': {'spotify': 'https://open.spotify.com/artist/3ipn9JLAPI5GUEo4y4jcoi'},
         'href': 'https://api.spotify.com/v1/artists/3ipn9JLAPI5GUEo4y4jcoi',
         'id': '3ipn9JLAPI5GUEo4y4jcoi',
         'name': 'Ludacris',
         'type': 'artist',
         'uri': 'spotify:artist:3ipn9JLAPI5GUEo4y4jcoi'}],
       'disc_number': 1,
       'duration_ms': 234893,
       'explicit': True,
       'external_ids': {'isrc': 'USEE10240932'},
       'external_urls': {'spotify': 'https://open.spotify.com/track/75DjPjiIp2fvJDjtt41Jfs'},
       'href': 'https://api.spotify.com/v1/tracks/75DjPjiIp2fvJDjtt41Jfs',
       'id': '75DjPjiIp2fvJDjtt41Jfs',
       'is_local': False,
       'is_playable': True,
       'name': 'Gossip Folks (feat. Ludacris)',
       'popularity': 53,
       'preview_url': 'https://p.scdn.co/mp3-preview/7fae13f01ac3dae5cb6dc419ceedf84a65f3f3ea?cid=e6ff82a6418a4191a5b3a95622faf5dd',
       'track_number': 3,
       'type': 'track',
       'uri': 'spotify:track:75DjPjiIp2fvJDjtt41Jfs'}]}



- Notes:
  - There may be high multicollinearity in these data, as these are a mix of primary and secondary information. For example, Spotify Web API explains that the danceability is based on other primary data including tempo, while the energy is based on some other primary data including loudness. Therefore, it would be important to find a method that are insensitive to the multicollinearity among columns.

### (3) Genius API

- Genius (https://genius.com/) is an American digital media library, particularly specialized for song lyrics and musical knowledge. Genius provides public Genius API where we can access lyrics data for songs.


- Below shows a screen capture of https://genius.com/Missy-elliott-lose-control-lyrics, which is the 0th track in the 0th playlist in the 0th file, along with the developer tool.
<img src="fig\Genius.png" width="635" height="460">


- Below shows a html-parsed result of this page by using `requests` and `BeautifulSoup`:



```python
GeniusURL = "https://genius.com/Missy-elliott-lose-control-lyrics"
BeautifulSoup(requests.get(GeniusURL).text,'html.parser')
```





    
    <!DOCTYPE html>
    
    <html class="snarly apple_music_player--enabled bagon_song_page--enabled song_stories_public_launch--enabled" lang="en" xml:lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:fb="http://www.facebook.com/2008/fbml">
    <head>
    <base href="//genius.com/" target="_top"/>
    <script type="text/javascript">
    //<![CDATA[
    
      var _sf_startpt=(new Date()).getTime();
      if (window.performance && performance.mark) {
        window.performance.mark('parse_start');
      }
    
    //]]>
    </script>
    <title>Missy Elliott – Lose Control Lyrics | Genius Lyrics</title>
    <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
    <meta content="width=device-width,initial-scale=1" name="viewport"/>
    <meta content="app-id=709482991" name="apple-itunes-app"/>
    <link href="https://assets.genius.com/images/apple-touch-icon.png?1544632201" rel="apple-touch-icon"/>
    <link href="https://assets.genius.com/images/apple-touch-icon.png?1544632201" rel="apple-touch-icon"/>
    <!-- Mobile IE allows us to activate ClearType technology for smoothing fonts for easy reading -->
    <meta content="on" http-equiv="cleartype"/>
    <meta content="f63347d284f184b0" name="y_key"/>
    <meta content="Genius" property="og:site_name">
    <meta content="265539304824" property="fb:app_id">
    <meta content="308252472676410" property="fb:pages">
    <link href="https://genius.com/opensearch.xml" rel="search" title="Genius" type="application/opensearchdescription+xml"/>
    <script>
      var CURRENT_USER = null;
      var CANONICAL_DOMAIN = "genius.com";
      var CANONICAL_DOMAIN_PARTS_LENGTH = 2;
      var CURRENT_TAG = {"tag":{"always_allow_personal_annotation":null,"beefy":true,"created_at":"2014-05-15T20:10:15Z","deleted_at":null,"description":"Rap Genius is dedicated to crowd-sourced (and [artist](http://rapgenius.com/Nas)/[producer](http://rapgenius.com/youngchopbeatz)-sourced) annotation of rap lyrics/[beats](/tags/producer-genius), from [\"Rapper's Delight\"](http://rapgenius.com/Sugar-hill-gang-rappers-delight-lyrics) to [*To Pimp A Butterfly.*](http://genius.com/Kendrick-lamar-to-pimp-a-butterfly-album-art-track-list-lyrics) Fan favorites [Rap Stats](http://genius.com/rapstats) & [The Rap Map](http://genius.com/map/) also call Rap Genius home. \r\n\r\nFind out all the latest on [Twitter](https://twitter.com/Genius) and [Facebook.](https://www.facebook.com/geniusdotcom)","designation":"featured_music","google_analytics_table_id":"ga:88245212","id":1434,"image_url":"https://images.rapgenius.com/922b8f934a36ecffb07c4cd83ecfbf2d.600x600x1.png","languages_store":{},"logical_name":"rap","music":true,"name":"Rap Genius","slug":"rap","taggings_count":892197,"twitter_name":"Genius","updated_at":"2018-12-12T17:29:38Z"}};
      var TRACKING_DATA = {"Song ID":33158,"Title":"Lose Control","Primary Artist":"Missy Elliott","Primary Artist ID":1529,"Primary Album":"The Cookbook","Primary Album ID":6618,"Tag":"rap","Primary Tag":"rap","Primary Tag ID":1434,"Music?":true,"Annotatable Type":"Song","Annotatable ID":33158,"featured_video":true,"cohort_ids":[],"has_verified_callout":false,"has_featured_annotation":true,"created_at":"2011-04-09T06:23:13Z","created_month":"2011-04-01","created_year":2011,"song_tier":"D","Has Recirculated Articles":true,"Lyrics Language":"en","Has Song Story":false,"Song Story ID":null,"Has Apple Match":true};
      var VALID_SUBDOMAINS = ["non-music","rap","rock","country","r-b","pop"];
      var EMBEDLY_KEY = "fc778e44915911e088ae4040f9f86dcd";
      var MOBILE_DEVICE = false;
      var APP_CONFIG = {"env":"production","api_root_url":"/api","transform_domain":"transform.genius.com","facebook_app_id":"265539304824","facebook_opengraph_api_version":"2.8","pusher_app_key":"6d893fcc6a0c695853ac","embedly_key":"fc778e44915911e088ae4040f9f86dcd","app_store_url":"https://itunes.apple.com/us/app/genius-by-rap-genius-search/id709482991?ls=1&mt=8","play_store_url":"https://play.google.com/store/apps/details?id=com.genius.android","soundcloud_client_id":"632c544d1c382f82526f369877aab5c0","annotator_context_length":32,"comment_reasons":[{"_type":"comment_reason","context_url":"https://genius.com/8846441/Genius-how-genius-works/More-on-annotations","display_character":"R","handle":"Restates the line","id":1,"name":"restates-the-line","raw_name":"restates the line","requires_body":false,"slug":"restates_the_line"},{"_type":"comment_reason","context_url":"https://genius.com/8846441/Genius-how-genius-works/More-on-annotations","display_character":"S","handle":"It’s a stretch","id":2,"name":"its-a-stretch","raw_name":"it’s a stretch","requires_body":false,"slug":"its_a_stretch"},{"_type":"comment_reason","context_url":"https://genius.com/8846441/Genius-how-genius-works/More-on-annotations","display_character":"M","handle":"Missing something","id":3,"name":"missing-something","raw_name":"missing something","requires_body":false,"slug":"missing_something"},{"_type":"comment_reason","context_url":"https://genius.com/8846441/Genius-how-genius-works/More-on-annotations","display_character":"…","handle":"Other","id":4,"name":"other","raw_name":"other","requires_body":true,"slug":"other"}],"comment_reasons_help_url":"https://genius.com/8846441/Genius-how-genius-works/More-on-annotations","filepicker_api_key":"Ar03MDs73TQm241ZgLwfjz","filepicker_policy":"eyJleHBpcnkiOjIzNTEwOTE1NTgsImNhbGwiOlsicGljayIsInJlYWQiLCJzdG9yZSIsInN0YXQiLCJjb252ZXJ0Il19","filepicker_signature":"68597b455e6c09bce0bfd73f758e299c95d49a5d5c8e808aaf4877da7801c4da","filepicker_s3_image_bucket":"filepicker-images-rapgenius","available_roles":["moderator","mega_boss","in_house_staff","verified_artist","meme_artist","engineer","editor","educator","staff","whitehat","tech_liaison","mediator"],"canonical_domain":"genius.com","enable_angular_debug":false,"fact_track_launch_article_url":"https://genius.com/a/genius-and-spotify-together","user_authority_roles":["moderator","editor","mediator"],"user_verification_roles":["community_artist","verified_artist","meme_artist"],"user_vote_types_for_delete":["votes","upvotes","downvotes"],"brightcove_account_id":"4863540648001","mixpanel_delayed_events_timeout":"86400","unreviewed_annotation_tooltip_info_url":"https://genius.com/8846524/Genius-how-genius-works/More-on-editorial-review","video_placements":{"desktop_song_page":[{"name":"sidebar","min_relevance":"high","fringe_min_relevance":"low","max_videos":0},{"name":"sidebar_thumb","min_relevance":"medium","fringe_min_relevance":"low","max_videos":0},{"name":"recirculated","min_relevance":"low","max_videos":3}],"mobile_song_page":[{"name":"footer","min_relevance":"medium","max_videos":1},{"name":"recirculated","min_relevance":"low","max_videos":3}],"desktop_artist_page":[{"name":"sidebar","min_relevance":"medium","fringe_min_relevance":"low","max_videos":2}],"mobile_artist_page":[{"name":"carousel","min_relevance":"medium","fringe_min_relevance":"low","max_videos":5}],"amp_song_page":[{"name":"footer","min_relevance":"medium","max_videos":1}],"amp_video_page":[{"name":"related","min_relevance":"low","max_videos":8}],"desktop_video_page":[{"name":"series_related","min_relevance":"low","max_videos":8,"series":true},{"name":"related","min_relevance":"low","max_videos":8}],"desktop_article_page":[{"name":"carousel","min_relevance":"low","max_videos":5}],"mobile_article_page":[{"name":"carousel","min_relevance":"low","max_videos":5}],"desktop_album_page":[{"name":"sidebar","min_relevance":"medium","fringe_min_relevance":"low","max_videos":1}],"amp_album_page":[{"name":"carousel","min_relevance":"low","max_videos":5}]},"app_name":"rapgenius-cedar","vttp_parner_id":"719c82b0-266e-11e7-827d-7f7dc47f6bc0","default_cover_art_url":"https://assets.genius.com/images/default_cover_art.png?1544632201","sizies_base_url":"https://t2.genius.com/unsafe","max_line_item_event_count":10,"dmp_match_threshold":0.05,"ab_tests":[],"external_song_match_purposes":["streaming_service_lyrics","streaming_service_player"],"brightcove_mobile_thumbnail_web_player_id":"SyGQSOxol","brightcove_modal_web_player_id":"S1LI5bh0","brightcove_song_story_web_player_id":"SkfSovRVf","brightcove_standard_web_player_id":"S1ZcmcOC1x","brightcove_standard_no_autoplay_web_player_id":"ByRtIUBvx","brightcove_sitemap_player_id":"BJfoOE1ol"};
      var SESSION_CONFIG = {"current_user":null,"taboola_enabled":false,"fringe_enabled":true,"log_client_metrics":false,"show_ads":true};
      var AD_CONFIG = {"ad_placements":{"amp_article_sticky":{"sizes":[[300,50],[320,50]],"kv":{"is_atf":false}},"amp_article_footer":{"sizes":[[300,250]],"kv":{"is_atf":false}},"amp_album_sticky":{"sizes":[[320,50]]},"amp_album_footer":{"sizes":[[300,250]]},"amp_song_annotation":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294868"}},{"bidder":"ix","params":{"id":10,"siteID":194719}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613918"}}]},"amp_song_below_player":{"sizes":[[300,250]],"kv":{"is_atf":false}},"amp_song_below_song_bio":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294916"}},{"bidder":"ix","params":{"id":8,"siteID":194716}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613910"}}]},"amp_song_leaderboard":{"sizes":[[320,50]],"kv":{"is_atf":true},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"13937404"}},{"bidder":"ix","params":{"id":25,"siteID":300787}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"1051268"}}]},"amp_song_medium1":{"sizes":[[300,250],[320,480]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294891"}},{"bidder":"ix","params":{"id":1,"siteID":194712}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613900"}}]},"amp_song_medium2":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294897"}},{"bidder":"ix","params":{"id":5,"siteID":194713}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613902"}}]},"amp_song_medium3":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294900"}},{"bidder":"ix","params":{"id":6,"siteID":194714}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613906"}}]},"amp_song_medium_footer":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294876"}},{"bidder":"ix","params":{"id":7,"siteID":194715}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613908"}}]},"amp_song_q_and_a":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294908"}},{"bidder":"ix","params":{"id":11,"siteID":194720}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613920"}}]},"amp_song_sticky":{"sizes":[[300,50],[320,50]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294881"}},{"bidder":"ix","params":{"id":3,"siteID":194711}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613898"}}]},"amp_video_leaderboard":{"sizes":[[320,50]]},"android_song_inread2":{"sizes":"MEDIUM_RECTANGLE"},"android_song_inread3":{"sizes":"MEDIUM_RECTANGLE"},"android_song_inread":{"sizes":"MEDIUM_RECTANGLE"},"app_song_inread2":{"sizes":"MEDIUM_RECTANGLE"},"app_song_inread3":{"sizes":"MEDIUM_RECTANGLE"},"app_song_inread":{"sizes":"MEDIUM_RECTANGLE"},"brightcove_article_web_player":{"sizes":[[640,360]]},"brightcove_modal_web_player":{"sizes":[[640,360]]},"brightcove_article_list_web_player":{"sizes":[[640,360]]},"brightcove_mobile_thumbnail_web_player":{"sizes":[[640,360]]},"desktop_album_leaderboard":{"sizes":[[970,250],[728,90],[970,90],[970,1]],"placeholder_size":[728,90],"kv":{"is_atf":true}},"desktop_album_sidebar":{"sizes":[[300,250]],"placeholder_size":[300,250],"kv":{"is_atf":true}},"desktop_article_sidebar":{"sizes":[[300,600],[300,250]]},"desktop_article_leaderboard":{"sizes":[[970,250],[728,90],[970,90],[970,1]],"placeholder_size":[728,90],"kv":{"is_atf":true}},"desktop_article_skin":{"sizes":[[1700,800],[1700,1200]],"kv":{"is_atf":true}},"desktop_artist_leaderboard":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_artist_sidebar":{"sizes":[[300,250]],"placeholder_size":[300,250]},"desktop_discussion_leaderboard":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_discussion_sidebar":{"sizes":[[300,250]],"placeholder_size":[300,250]},"desktop_forum_leaderboard":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_forum_medium1":{"sizes":[[300,250]]},"desktop_home_footer":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_home_leaderboard":{"sizes":[[970,250],[728,90],[970,1]],"kv":{"is_atf":true}},"desktop_home_medium1":{"sizes":[[300,600]],"kv":{"is_atf":false}},"desktop_home_skin":{"sizes":[[1700,800],[1700,1200]],"kv":{"is_atf":true}},"desktop_search_leaderboard":{"sizes":[[970,250],[728,90]],"placeholder_size":[728,90]},"desktop_search_sidebar":{"sizes":[[300,250],[300,600]]},"desktop_song_annotation":{"sizes":[[300,250]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294796"}},{"bidder":"ix","params":{"siteId":194718,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639856"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"399321791"}}]},"annotation_open_mobile":{"sizes":[[1,1]],"kv":{"is_atf":false}},"desktop_song_comments":{"sizes":[[300,250],[336,280],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294806"}},{"bidder":"ix","params":{"siteId":194722,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639858"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703511"}}]},"desktop_song_inread":{"sizes":[[1,1],[1,2],[300,250],[336,280],[320,480],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11476342"}},{"bidder":"ix","params":{"siteId":201300,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"660418"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"351297791"}}]},"desktop_song_inread2":{"sizes":[[300,250],[336,280],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11476344"}},{"bidder":"ix","params":{"siteId":201301,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"660420"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"464408951"}}]},"desktop_song_inread3":{"sizes":[[300,250],[336,280],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11476345"}},{"bidder":"ix","params":{"id":21,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"660424"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"464409071"}}]},"desktop_song_leaderboard":{"sizes":[[970,250],[728,90],[970,90],[970,1]],"placeholder_size":[728,90],"kv":{"is_atf":true},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294811"}},{"bidder":"ix","params":{"siteId":197229,"size":[728,90]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639864"}}]},"desktop_song_lyrics_footer":{"sizes":[[300,250],[300,600],[336,280],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294823"}},{"bidder":"ix","params":{"siteId":194710,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639860"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"397849871"}}]},"desktop_song_medium1":{"sizes":[[300,600],[300,250]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294830"}},{"bidder":"ix","params":{"siteId":194709,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639862"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"346199471"}}]},"desktop_song_out_of_page":{"out_of_page":true},"desktop_song_q_and_a":{"sizes":[[300,250]],"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11474887"}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"660286"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"464408711"}}]},"desktop_song_sidebar_top":{"sizes":[[300,250]],"placeholder_size":[300,250],"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11621942"}},{"bidder":"ix","params":{"siteId":221945,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"682610"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"21615783137"}}]},"desktop_song_skin":{"sizes":[[1700,800],[1700,1200]],"kv":{"is_atf":true}},"desktop_user_leaderboard":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_user_sidebar":{"sizes":[[300,250]],"placeholder_size":[300,250]},"desktop_video_sidebar":{"sizes":[[300,250],[300,600]],"placeholder_size":[300,250]},"mobile_artist_leaderboard":{"sizes":[[320,50]],"placeholder_size":[320,50]},"mobile_artist_footer":{"sizes":[[300,250]],"placeholder_size":[300,250]},"mobile_discussion_leaderboard":{"sizes":[[320,50]],"placeholder_size":[320,50]},"mobile_home_adhesion":{"sizes":[[320,50],[320,100]],"kv":{"is_atf":true}},"mobile_home_footer":{"sizes":[[300,250]],"kv":{"is_atf":false}},"mobile_forum_leaderboard":{"sizes":[[320,50]],"placeholder_size":[320,50]},"mobile_search_leaderboard":{"sizes":[[320,50]],"placeholder_size":[320,50]},"mobile_search_footer":{"sizes":[[300,250]],"placeholder_size":[300,250]},"mobile_song_annotation":{"sizes":[[300,250]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294868"}},{"bidder":"ix","params":{"siteId":194719,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613918"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"399321431"}}]},"mobile_song_comments":{"sizes":[[300,250],[300,50],[320,50],[320,100]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294870"}},{"bidder":"ix","params":{"siteId":194721,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613922"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703391"}}]},"mobile_song_footer":{"sizes":[[300,250],[300,50],[320,50],[320,100]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294876"}},{"bidder":"ix","params":{"siteId":194715,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613908"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"395129471"}}]},"mobile_song_interstitial":{"sizes":[[1,1]],"kv":{"is_atf":false}},"mobile_song_lyrics_header":{"sizes":[[320,50],[320,100]],"kv":{"is_atf":true},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"13937403"}},{"bidder":"ix","params":{"id":26,"siteID":300788}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"1051274"}}]},"mobile_song_lyrics_header_adhesion":{"sizes":[[300,50],[320,50],[320,100]],"kv":{"is_atf":true},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294881"}},{"bidder":"ix","params":{"siteId":194711,"size":[300,50]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613898"}}]},"mobile_song_medium1":{"sizes":[[1,1],[1,2],[300,250],[320,480]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294891"}},{"bidder":"ix","params":{"siteId":194712,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613900"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"346200911"}}]},"mobile_song_medium2":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294897"}},{"bidder":"ix","params":{"siteId":194713,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613902"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"346201631"}}]},"mobile_song_medium3":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294900"}},{"bidder":"ix","params":{"siteId":194714,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613906"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703151"}}]},"mobile_song_out_of_page":{"out_of_page":true},"mobile_song_q_and_a":{"sizes":[[300,250],[300,50],[320,50],[320,100]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294908"}},{"bidder":"ix","params":{"siteId":194720,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613920"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703271"}}]},"mobile_song_song_bio":{"sizes":[[300,250],[300,50],[320,50],[320,100]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294916"}},{"bidder":"ix","params":{"siteId":194716,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613910"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703991"}}]},"mobile_user_leaderboard":{"sizes":[[320,50]]},"mobile_user_footer":{"sizes":[[300,250]]},"web_annotator_annotation":{"sizes":[[300,250]],"kv":{"is_atf":false}}},"dfp_network_id":"342026871","a9_pub_id":"3459","prebid":{"priceGranularity":{"buckets":[{"min":0.1,"max":0.99,"increment":0.01},{"min":1.0,"max":4.98,"increment":0.02},{"min":5.0,"max":20.0,"increment":0.1}]},"bidderSettings":{"districtmDMX":{"bidCpmAdjustment":0.9}},"sizes":[[970,250],[970,90],[728,90],[468,60],[336,280],[300,50],[300,250],[300,600],[320,100],[320,50],[1,2]],"experimental_bidders":[]},"platform":"desktop","prebid_timeout":1000,"prebid_terminal_timeout":1500,"header_bidding_enabled":true,"prebid_server_enabled":false,"prebid_server_timeout":400,"prebid_server":{"appnexus":{"account_id":"c438d9e0-0182-4737-8ad7-68bcaf49d76e","endpoint":"https://prebid.adnxs.com/pbs/v1/auction","sync_endpoint":"https://prebid.adnxs.com/pbs/v1/cookie_sync"},"rubicon":{"account_id":"15874","endpoint":"https://prebid-server.rubiconproject.com/auction","sync_endpoint":"https://prebid-server.rubiconproject.com/cookie_sync"}},"ias_enabled":true,"ias_pubid":"927569","cmp_enabled":false,"consent_timeout":10000}
      var ANALYTICS_CONFIG = {"enabled":true,"mixpanel_enabled":true,"comscore_client_id":"17151659","quantcast_account":"p-f3CPQ6vHckedE","librato_web_client_host":"librato-collector.genius.com"}
      var TOP_LEVEL_BLOCK_CONTAINERS = ["address","article","aside","blockquote","div","dl","fieldset","footer","form","h1","h2","h3","h4","h5","h6","header","menu","nav","ol","p","pre","section","table","ul"];
      var TOP_LEVEL_STANDALONE_BLOCKS = ["img","hr"];
    </script>
    <script async="" src="https://www.youtube.com/iframe_api"></script>
    <script defer="" src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
    <script>
      window['Genius.cmp'] = window['Genius.cmp'] || [];
    </script>
    <!--sse-->
    <script>window['Genius.ads'] = window['Genius.ads'] || [];</script>
    <script defer="true" src="https://assets.genius.com/javascripts/compiled/ads-db9d43845ac56f5e603b.js" type="text/javascript"></script>
    <script async="true" src="https://www.googletagservices.com/tag/js/gpt.js" type="text/javascript"></script>
    <script>
        window['Genius.ads'].push(function(ads) {
          var config = {"ad_placements":{"amp_article_sticky":{"sizes":[[300,50],[320,50]],"kv":{"is_atf":false}},"amp_article_footer":{"sizes":[[300,250]],"kv":{"is_atf":false}},"amp_album_sticky":{"sizes":[[320,50]]},"amp_album_footer":{"sizes":[[300,250]]},"amp_song_annotation":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294868"}},{"bidder":"ix","params":{"id":10,"siteID":194719}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613918"}}]},"amp_song_below_player":{"sizes":[[300,250]],"kv":{"is_atf":false}},"amp_song_below_song_bio":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294916"}},{"bidder":"ix","params":{"id":8,"siteID":194716}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613910"}}]},"amp_song_leaderboard":{"sizes":[[320,50]],"kv":{"is_atf":true},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"13937404"}},{"bidder":"ix","params":{"id":25,"siteID":300787}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"1051268"}}]},"amp_song_medium1":{"sizes":[[300,250],[320,480]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294891"}},{"bidder":"ix","params":{"id":1,"siteID":194712}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613900"}}]},"amp_song_medium2":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294897"}},{"bidder":"ix","params":{"id":5,"siteID":194713}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613902"}}]},"amp_song_medium3":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294900"}},{"bidder":"ix","params":{"id":6,"siteID":194714}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613906"}}]},"amp_song_medium_footer":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294876"}},{"bidder":"ix","params":{"id":7,"siteID":194715}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613908"}}]},"amp_song_q_and_a":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294908"}},{"bidder":"ix","params":{"id":11,"siteID":194720}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613920"}}]},"amp_song_sticky":{"sizes":[[300,50],[320,50]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294881"}},{"bidder":"ix","params":{"id":3,"siteID":194711}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613898"}}]},"amp_video_leaderboard":{"sizes":[[320,50]]},"android_song_inread2":{"sizes":"MEDIUM_RECTANGLE"},"android_song_inread3":{"sizes":"MEDIUM_RECTANGLE"},"android_song_inread":{"sizes":"MEDIUM_RECTANGLE"},"app_song_inread2":{"sizes":"MEDIUM_RECTANGLE"},"app_song_inread3":{"sizes":"MEDIUM_RECTANGLE"},"app_song_inread":{"sizes":"MEDIUM_RECTANGLE"},"brightcove_article_web_player":{"sizes":[[640,360]]},"brightcove_modal_web_player":{"sizes":[[640,360]]},"brightcove_article_list_web_player":{"sizes":[[640,360]]},"brightcove_mobile_thumbnail_web_player":{"sizes":[[640,360]]},"desktop_album_leaderboard":{"sizes":[[970,250],[728,90],[970,90],[970,1]],"placeholder_size":[728,90],"kv":{"is_atf":true}},"desktop_album_sidebar":{"sizes":[[300,250]],"placeholder_size":[300,250],"kv":{"is_atf":true}},"desktop_article_sidebar":{"sizes":[[300,600],[300,250]]},"desktop_article_leaderboard":{"sizes":[[970,250],[728,90],[970,90],[970,1]],"placeholder_size":[728,90],"kv":{"is_atf":true}},"desktop_article_skin":{"sizes":[[1700,800],[1700,1200]],"kv":{"is_atf":true}},"desktop_artist_leaderboard":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_artist_sidebar":{"sizes":[[300,250]],"placeholder_size":[300,250]},"desktop_discussion_leaderboard":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_discussion_sidebar":{"sizes":[[300,250]],"placeholder_size":[300,250]},"desktop_forum_leaderboard":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_forum_medium1":{"sizes":[[300,250]]},"desktop_home_footer":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_home_leaderboard":{"sizes":[[970,250],[728,90],[970,1]],"kv":{"is_atf":true}},"desktop_home_medium1":{"sizes":[[300,600]],"kv":{"is_atf":false}},"desktop_home_skin":{"sizes":[[1700,800],[1700,1200]],"kv":{"is_atf":true}},"desktop_search_leaderboard":{"sizes":[[970,250],[728,90]],"placeholder_size":[728,90]},"desktop_search_sidebar":{"sizes":[[300,250],[300,600]]},"desktop_song_annotation":{"sizes":[[300,250]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294796"}},{"bidder":"ix","params":{"siteId":194718,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639856"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"399321791"}}]},"annotation_open_mobile":{"sizes":[[1,1]],"kv":{"is_atf":false}},"desktop_song_comments":{"sizes":[[300,250],[336,280],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294806"}},{"bidder":"ix","params":{"siteId":194722,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639858"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703511"}}]},"desktop_song_inread":{"sizes":[[1,1],[1,2],[300,250],[336,280],[320,480],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11476342"}},{"bidder":"ix","params":{"siteId":201300,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"660418"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"351297791"}}]},"desktop_song_inread2":{"sizes":[[300,250],[336,280],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11476344"}},{"bidder":"ix","params":{"siteId":201301,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"660420"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"464408951"}}]},"desktop_song_inread3":{"sizes":[[300,250],[336,280],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11476345"}},{"bidder":"ix","params":{"id":21,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"660424"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"464409071"}}]},"desktop_song_leaderboard":{"sizes":[[970,250],[728,90],[970,90],[970,1]],"placeholder_size":[728,90],"kv":{"is_atf":true},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294811"}},{"bidder":"ix","params":{"siteId":197229,"size":[728,90]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639864"}}]},"desktop_song_lyrics_footer":{"sizes":[[300,250],[300,600],[336,280],[468,60]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294823"}},{"bidder":"ix","params":{"siteId":194710,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639860"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"397849871"}}]},"desktop_song_medium1":{"sizes":[[300,600],[300,250]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294830"}},{"bidder":"ix","params":{"siteId":194709,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"639862"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"346199471"}}]},"desktop_song_out_of_page":{"out_of_page":true},"desktop_song_q_and_a":{"sizes":[[300,250]],"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11474887"}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"660286"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"464408711"}}]},"desktop_song_sidebar_top":{"sizes":[[300,250]],"placeholder_size":[300,250],"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11621942"}},{"bidder":"ix","params":{"siteId":221945,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"135638","zoneId":"682610"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"21615783137"}}]},"desktop_song_skin":{"sizes":[[1700,800],[1700,1200]],"kv":{"is_atf":true}},"desktop_user_leaderboard":{"sizes":[[728,90]],"placeholder_size":[728,90]},"desktop_user_sidebar":{"sizes":[[300,250]],"placeholder_size":[300,250]},"desktop_video_sidebar":{"sizes":[[300,250],[300,600]],"placeholder_size":[300,250]},"mobile_artist_leaderboard":{"sizes":[[320,50]],"placeholder_size":[320,50]},"mobile_artist_footer":{"sizes":[[300,250]],"placeholder_size":[300,250]},"mobile_discussion_leaderboard":{"sizes":[[320,50]],"placeholder_size":[320,50]},"mobile_home_adhesion":{"sizes":[[320,50],[320,100]],"kv":{"is_atf":true}},"mobile_home_footer":{"sizes":[[300,250]],"kv":{"is_atf":false}},"mobile_forum_leaderboard":{"sizes":[[320,50]],"placeholder_size":[320,50]},"mobile_search_leaderboard":{"sizes":[[320,50]],"placeholder_size":[320,50]},"mobile_search_footer":{"sizes":[[300,250]],"placeholder_size":[300,250]},"mobile_song_annotation":{"sizes":[[300,250]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294868"}},{"bidder":"ix","params":{"siteId":194719,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613918"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"399321431"}}]},"mobile_song_comments":{"sizes":[[300,250],[300,50],[320,50],[320,100]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294870"}},{"bidder":"ix","params":{"siteId":194721,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613922"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703391"}}]},"mobile_song_footer":{"sizes":[[300,250],[300,50],[320,50],[320,100]],"placeholder_size":[300,250],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294876"}},{"bidder":"ix","params":{"siteId":194715,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613908"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"395129471"}}]},"mobile_song_interstitial":{"sizes":[[1,1]],"kv":{"is_atf":false}},"mobile_song_lyrics_header":{"sizes":[[320,50],[320,100]],"kv":{"is_atf":true},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"13937403"}},{"bidder":"ix","params":{"id":26,"siteID":300788}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"1051274"}}]},"mobile_song_lyrics_header_adhesion":{"sizes":[[300,50],[320,50],[320,100]],"kv":{"is_atf":true},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294881"}},{"bidder":"ix","params":{"siteId":194711,"size":[300,50]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613898"}}]},"mobile_song_medium1":{"sizes":[[1,1],[1,2],[300,250],[320,480]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294891"}},{"bidder":"ix","params":{"siteId":194712,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613900"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"346200911"}}]},"mobile_song_medium2":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294897"}},{"bidder":"ix","params":{"siteId":194713,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613902"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"346201631"}}]},"mobile_song_medium3":{"sizes":[[300,250]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294900"}},{"bidder":"ix","params":{"siteId":194714,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613906"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703151"}}]},"mobile_song_out_of_page":{"out_of_page":true},"mobile_song_q_and_a":{"sizes":[[300,250],[300,50],[320,50],[320,100]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294908"}},{"bidder":"ix","params":{"siteId":194720,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613920"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703271"}}]},"mobile_song_song_bio":{"sizes":[[300,250],[300,50],[320,50],[320,100]],"kv":{"is_atf":false},"prebid_placements":[{"bidder":"appnexus","params":{"placementId":"11294916"}},{"bidder":"ix","params":{"siteId":194716,"size":[300,250]}},{"bidder":"rubicon","params":{"accountId":"15874","siteId":"107816","zoneId":"613910"}},{"bidder":"undertone","params":{"publisherId":"3603","placementId":"408703991"}}]},"mobile_user_leaderboard":{"sizes":[[320,50]]},"mobile_user_footer":{"sizes":[[300,250]]},"web_annotator_annotation":{"sizes":[[300,250]],"kv":{"is_atf":false}}},"dfp_network_id":"342026871","a9_pub_id":"3459","prebid":{"priceGranularity":{"buckets":[{"min":0.1,"max":0.99,"increment":0.01},{"min":1.0,"max":4.98,"increment":0.02},{"min":5.0,"max":20.0,"increment":0.1}]},"bidderSettings":{"districtmDMX":{"bidCpmAdjustment":0.9}},"sizes":[[970,250],[970,90],[728,90],[468,60],[336,280],[300,50],[300,250],[300,600],[320,100],[320,50],[1,2]],"experimental_bidders":[]},"platform":"desktop","prebid_timeout":1000,"prebid_terminal_timeout":1500,"header_bidding_enabled":true,"prebid_server_enabled":false,"prebid_server_timeout":400,"prebid_server":{"appnexus":{"account_id":"c438d9e0-0182-4737-8ad7-68bcaf49d76e","endpoint":"https://prebid.adnxs.com/pbs/v1/auction","sync_endpoint":"https://prebid.adnxs.com/pbs/v1/cookie_sync"},"rubicon":{"account_id":"15874","endpoint":"https://prebid-server.rubiconproject.com/auction","sync_endpoint":"https://prebid-server.rubiconproject.com/cookie_sync"}},"ias_enabled":true,"ias_pubid":"927569","cmp_enabled":false,"consent_timeout":10000};
          var targeting_list = [{"name":"song_id","values":["33158"]},{"name":"song_title","values":["Lose Control"]},{"name":"artist_id","values":["1529"]},{"name":"artist_name","values":["Missy Elliott"]},{"name":"is_explicit","values":["true"]},{"name":"pageviews","values":["55627"]},{"name":"primary_tag_id","values":["1434"]},{"name":"tag_id","values":["1434"]},{"name":"song_tier","values":["D"]},{"name":"topic","values":[]},{"name":"has_song_story","values":["false"]},{"name":"in_top_10","values":["false"]},{"name":"artist_in_top_10","values":["false"]},{"name":"album_in_top_10","values":["false"]},{"name":"new_release","values":["false"]},{"name":"release_month","values":["200505"]},{"name":"release_year","values":["2005"]},{"name":"release_decade","values":["2000"]},{"name":"in_top_10_rap","values":["false"]},{"name":"in_top_10_rock","values":["false"]},{"name":"in_top_10_country","values":["false"]},{"name":"in_top_10_r_and_b","values":["false"]},{"name":"in_top_10_pop","values":["false"]},{"name":"template","values":["song"]},{"name":"environment","values":["production"]},{"name":"platform","values":["web"]}];
          var targeting = {};
          targeting_list.forEach(function(pair) {
            targeting[pair.name] = pair.values;
          });
    
          ads.initialize({config: config, targeting: targeting});
          ads.ensure_targeting_for_placements([["desktop_song_leaderboard","desktop_song_leaderboard"],["desktop_song_sidebar_top","desktop_song_sidebar_top"],["desktop_song_medium1","song_page_sidebar"]]);
        });
      </script>
    <script async="" src="https://cdn.adsafeprotected.com/iasPET.1.js"></script>
    <!--/sse-->
    <link href="https://assets.genius.com/stylesheets/compiled/bagon_desktop-da111cdd762a0d9c32886eb27984f172.css" media="screen" rel="stylesheet" type="text/css">
    <link as="script" href="https://assets.genius.com/javascripts/compiled/bagon_desktop-db9d43845ac56f5e603b.js" rel="preload"/><script defer="true" src="https://assets.genius.com/javascripts/compiled/bagon_desktop-db9d43845ac56f5e603b.js" type="text/javascript"></script>
    <style>
      @font-face {
        font-family: 'Programme';
        src: url(data:font/woff2;base64,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) format('woff2'),
          url(https://assets.genius.com/fonts/programme_bold.woff?1544632201) format('woff');
        font-style: normal;
        font-weight: bold;
      }
    
      @font-face {
        font-family: 'Programme';
        src: url(https://assets.genius.com/fonts/programme_normal.woff2?1544632201) format('woff2'),
          url(https://assets.genius.com/fonts/programme_normal.woff?1544632201) format('woff');
        font-style: normal;
        font-weight: normal;
      }
    
      @font-face {
        font-family: 'Programme';
        src: url(https://assets.genius.com/fonts/programme_normal_italic.woff2?1544632201) format('woff2'),
          url(https://assets.genius.com/fonts/programme_normal_italic.woff?1544632201) format('woff');
        font-style: italic;
        font-weight: normal;
      }
    
      @font-face {
        font-family: 'Programme';
        src: url(https://assets.genius.com/fonts/programme_light.woff2?1544632201) format('woff2'),
          url(https://assets.genius.com/fonts/programme_light.woff?1544632201) format('woff');
        font-style: normal;
        font-weight: 100;
      }
    
      @font-face {
        font-family: 'Programme';
        src: url(https://assets.genius.com/fonts/programme_light_italic.woff2?1544632201) format('woff2'),
          url(https://assets.genius.com/fonts/programme_light_italic.woff?1544632201) format('woff');
        font-style: italic;
        font-weight: 100;
      }
    </style>
    <meta content="https://genius.com/Missy-elliott-lose-control-lyrics" property="og:url">
    <meta content="music.song" property="og:type"/>
    <meta content="Missy Elliott (Ft. Ciara &amp; Fatman Scoop) – Lose Control" property="og:title"/>
    <meta content="Another classic Missy Elliott banger, centred around the loss of control of your extremities on the dance floor. It samples Hot Streak’s “Body Work” and Cybotron’s “Clear,” and" property="og:description"/>
    <meta content="https://images.genius.com/51639e476eac5cb5bed5794c023763ff.700x707x1.jpg" property="og:image"/>
    <meta content="https://genius.com/Missy-elliott-lose-control-lyrics" property="twitter:url"/>
    <meta content="music.song" property="twitter:type"/>
    <meta content="Missy Elliott (Ft. Ciara &amp; Fatman Scoop) – Lose Control" property="twitter:title"/>
    <meta content="Another classic Missy Elliott banger, centred around the loss of control of your extremities on the dance floor. It samples Hot Streak’s “Body Work” and Cybotron’s “Clear,” and" property="twitter:description"/>
    <meta content="https://images.genius.com/51639e476eac5cb5bed5794c023763ff.700x707x1.jpg" property="twitter:image"/>
    <meta content="@Genius" property="twitter:site"/>
    <meta content="summary_large_image" property="twitter:card"/>
    <meta content="Genius" property="twitter:app:name:iphone"/>
    <meta content="709482991" property="twitter:app:id:iphone"/>
    <meta content="genius://songs/33158" property="twitter:app:url:iphone"/>
    <meta content="Lose Control Lyrics: Music make you lose control, music make you lose control / Let's go! Hey, hey, hey, hey, hey, hey / Here we go now, here we go now, here we go now, here we go now / (Music make you" name="description"/>
    <link href="ios-app://709482991/genius/songs/33158" rel="alternate"/>
    <meta content="/songs/33158" name="newrelic-resource-path"/>
    <link href="https://genius.com/Missy-elliott-lose-control-lyrics" rel="canonical"/>
    <link href="https://genius.com/amp/Missy-elliott-lose-control-lyrics" rel="amphtml"/>
    <script type="text/javascript">
      var _qevents = _qevents || [];
      (function() {
        var elem = document.createElement('script');
        elem.src = (document.location.protocol == 'https:' ? 'https://secure' : 'http://edge') + '.quantserve.com/quant.js';
        elem.async = true;
        elem.type = 'text/javascript';
        var scpt = document.getElementsByTagName('script')[0];
        scpt.parentNode.insertBefore(elem, scpt);
      })();
    </script>
    <script type="text/javascript">
      window.ga = window.ga || function() {
        (window.ga.q = window.ga.q || []).push(arguments);
      };
    
      
        (function(g, e, n, i, u, s) {
          g['GoogleAnalyticsObject'] = 'ga';
          g.ga.l = Date.now();
          u = e.createElement(n);
          s = e.getElementsByTagName(n)[0];
          u.async = true;
          u.src = i;
          s.parentNode.insertBefore(u, s);
        })(window, document, 'script', 'https://www.google-analytics.com/analytics.js');
    
        ga('create', "UA-10346621-1", 'auto', {'useAmpClientId': true});
        ga('set', 'dimension1', "false");
        ga('set', 'dimension2', "songs#show");
        ga('set', 'dimension3', "rap");
        ga('set', 'dimension4', "true");
        ga('set', 'dimension5', 'false');
        ga('set', 'dimension6', 'none');
        ga('send', 'pageview');
      
    </script>
    <meta content="{&quot;chartbeat&quot;:{&quot;authors&quot;:&quot;Missy Elliott,Ciara,Fatman Scoop&quot;,&quot;sections&quot;:&quot;songs,tag:rap&quot;,&quot;title&quot;:&quot;Missy Elliott – Lose Control Lyrics | Genius Lyrics&quot;},&quot;controller_and_action&quot;:&quot;songs#show&quot;,&quot;dmp_data_layer&quot;:{&quot;page&quot;:{&quot;type&quot;:&quot;song&quot;,&quot;artists&quot;:[&quot;Missy Elliott&quot;,&quot;Ciara&quot;,&quot;Fatman Scoop&quot;],&quot;artist_ids&quot;:[&quot;[1529]&quot;,&quot;[1630]&quot;,&quot;[3516]&quot;],&quot;albums&quot;:[&quot;The Cookbook&quot;,&quot;Respect M.E.&quot;],&quot;album_ids&quot;:[&quot;[6618]&quot;,&quot;[335070]&quot;],&quot;genres&quot;:[&quot;Rap Genius&quot;],&quot;genre_ids&quot;:[&quot;[1434]&quot;],&quot;in_top_10&quot;:false,&quot;artist_in_top_10&quot;:false,&quot;album_in_top_10&quot;:false,&quot;new_release&quot;:false,&quot;release_month&quot;:&quot;200505&quot;,&quot;release_year&quot;:2005,&quot;release_decade&quot;:2000}},&quot;header_bid_placements&quot;:[[&quot;desktop_song_leaderboard&quot;,&quot;desktop_song_leaderboard&quot;],[&quot;desktop_song_sidebar_top&quot;,&quot;desktop_song_sidebar_top&quot;],[&quot;desktop_song_medium1&quot;,&quot;song_page_sidebar&quot;]],&quot;page_type&quot;:&quot;song&quot;,&quot;path&quot;:&quot;/Missy-elliott-lose-control-lyrics&quot;,&quot;probably_spam&quot;:false,&quot;title&quot;:&quot;Missy Elliott – Lose Control Lyrics | Genius Lyrics&quot;,&quot;tracking_data&quot;:[{&quot;key&quot;:&quot;Song ID&quot;,&quot;value&quot;:33158},{&quot;key&quot;:&quot;Title&quot;,&quot;value&quot;:&quot;Lose Control&quot;},{&quot;key&quot;:&quot;Primary Artist&quot;,&quot;value&quot;:&quot;Missy Elliott&quot;},{&quot;key&quot;:&quot;Primary Artist ID&quot;,&quot;value&quot;:1529},{&quot;key&quot;:&quot;Primary Album&quot;,&quot;value&quot;:&quot;The Cookbook&quot;},{&quot;key&quot;:&quot;Primary Album ID&quot;,&quot;value&quot;:6618},{&quot;key&quot;:&quot;Tag&quot;,&quot;value&quot;:&quot;rap&quot;},{&quot;key&quot;:&quot;Primary Tag&quot;,&quot;value&quot;:&quot;rap&quot;},{&quot;key&quot;:&quot;Primary Tag ID&quot;,&quot;value&quot;:1434},{&quot;key&quot;:&quot;Music?&quot;,&quot;value&quot;:true},{&quot;key&quot;:&quot;Annotatable Type&quot;,&quot;value&quot;:&quot;Song&quot;},{&quot;key&quot;:&quot;Annotatable ID&quot;,&quot;value&quot;:33158},{&quot;key&quot;:&quot;featured_video&quot;,&quot;value&quot;:true},{&quot;key&quot;:&quot;cohort_ids&quot;,&quot;value&quot;:[]},{&quot;key&quot;:&quot;has_verified_callout&quot;,&quot;value&quot;:false},{&quot;key&quot;:&quot;has_featured_annotation&quot;,&quot;value&quot;:true},{&quot;key&quot;:&quot;created_at&quot;,&quot;value&quot;:&quot;2011-04-09T06:23:13Z&quot;},{&quot;key&quot;:&quot;created_month&quot;,&quot;value&quot;:&quot;2011-04-01&quot;},{&quot;key&quot;:&quot;created_year&quot;,&quot;value&quot;:2011},{&quot;key&quot;:&quot;song_tier&quot;,&quot;value&quot;:&quot;D&quot;},{&quot;key&quot;:&quot;Has Recirculated Articles&quot;,&quot;value&quot;:true},{&quot;key&quot;:&quot;Lyrics Language&quot;,&quot;value&quot;:&quot;en&quot;},{&quot;key&quot;:&quot;Has Song Story&quot;,&quot;value&quot;:false},{&quot;key&quot;:&quot;Song Story ID&quot;,&quot;value&quot;:null},{&quot;key&quot;:&quot;Has Apple Match&quot;,&quot;value&quot;:true}],&quot;answered_pending_question_count&quot;:0,&quot;dfp_kv&quot;:[{&quot;name&quot;:&quot;song_id&quot;,&quot;values&quot;:[&quot;33158&quot;]},{&quot;name&quot;:&quot;song_title&quot;,&quot;values&quot;:[&quot;Lose Control&quot;]},{&quot;name&quot;:&quot;artist_id&quot;,&quot;values&quot;:[&quot;1529&quot;]},{&quot;name&quot;:&quot;artist_name&quot;,&quot;values&quot;:[&quot;Missy Elliott&quot;]},{&quot;name&quot;:&quot;is_explicit&quot;,&quot;values&quot;:[&quot;true&quot;]},{&quot;name&quot;:&quot;pageviews&quot;,&quot;values&quot;:[&quot;55627&quot;]},{&quot;name&quot;:&quot;primary_tag_id&quot;,&quot;values&quot;:[&quot;1434&quot;]},{&quot;name&quot;:&quot;tag_id&quot;,&quot;values&quot;:[&quot;1434&quot;]},{&quot;name&quot;:&quot;song_tier&quot;,&quot;values&quot;:[&quot;D&quot;]},{&quot;name&quot;:&quot;topic&quot;,&quot;values&quot;:[]},{&quot;name&quot;:&quot;has_song_story&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;in_top_10&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;artist_in_top_10&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;album_in_top_10&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;new_release&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;release_month&quot;,&quot;values&quot;:[&quot;200505&quot;]},{&quot;name&quot;:&quot;release_year&quot;,&quot;values&quot;:[&quot;2005&quot;]},{&quot;name&quot;:&quot;release_decade&quot;,&quot;values&quot;:[&quot;2000&quot;]},{&quot;name&quot;:&quot;in_top_10_rap&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;in_top_10_rock&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;in_top_10_country&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;in_top_10_r_and_b&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;in_top_10_pop&quot;,&quot;values&quot;:[&quot;false&quot;]},{&quot;name&quot;:&quot;template&quot;,&quot;values&quot;:[&quot;song&quot;]},{&quot;name&quot;:&quot;environment&quot;,&quot;values&quot;:[&quot;production&quot;]},{&quot;name&quot;:&quot;platform&quot;,&quot;values&quot;:[&quot;web&quot;]}],&quot;pending_question_count&quot;:0,&quot;show_edit_form&quot;:false,&quot;show_featured_question&quot;:false,&quot;spotify_referral&quot;:false,&quot;annotation_previews&quot;:[],&quot;default_questions&quot;:[],&quot;featured_question&quot;:null,&quot;lyrics_data&quot;:{&quot;client_timestamps&quot;:{&quot;lyrics_updated_at&quot;:1543136767,&quot;updated_by_human_at&quot;:1543136767},&quot;body&quot;:{&quot;html&quot;:&quot;&lt;p&gt;[Intro: Fatman Scoop]&lt;br&gt;\n&lt;a href=\&quot;/Missy-elliott-lose-control-lyrics#note-1760210\&quot; data-id=\&quot;1760210\&quot; class=\&quot;referent\&quot; ng-click=\&quot;open()\&quot; ng-class=\&quot;{\n          'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),\n          'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),\n          'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)\n        }\&quot; prevent-default-click=\&quot;\&quot; annotation-fragment=\&quot;1760210\&quot; on-hover-with-no-digest=\&quot;set_current_hover_and_digest(hover ? fragment_id : undefined)\&quot; classification=\&quot;accepted\&quot; image=\&quot;false\&quot; pending-editorial-actions-count=\&quot;0\&quot;&gt;&lt;i&gt;Music make you lose control, music make you lose control&lt;/i&gt;&lt;/a&gt;&lt;br&gt;\nLet's go! Hey, hey, hey, hey, hey, hey&lt;br&gt;\nHere we go now, here we go now, here we go now, here we go now&lt;br&gt;\n&lt;a href=\&quot;/Missy-elliott-lose-control-lyrics#note-8681095\&quot; data-id=\&quot;8681095\&quot; class=\&quot;referent\&quot; ng-click=\&quot;open()\&quot; ng-class=\&quot;{\n          'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),\n          'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),\n          'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)\n        }\&quot; prevent-default-click=\&quot;\&quot; annotation-fragment=\&quot;8681095\&quot; on-hover-with-no-digest=\&quot;set_current_hover_and_digest(hover ? fragment_id : undefined)\&quot; classification=\&quot;accepted\&quot; image=\&quot;false\&quot; pending-editorial-actions-count=\&quot;0\&quot;&gt;(Music make you lose control)&lt;/a&gt;&lt;br&gt;\nMisdemeanor's in the house&lt;br&gt;\nCiara's in the house&lt;br&gt;\nMisdemeanor's in the house&lt;br&gt;\nFatman Scoop-man Scoop-man Scoop..&lt;br&gt;\n&lt;br&gt;\n[Verse 1: Missy]&lt;br&gt;\n&lt;a href=\&quot;/Missy-elliott-lose-control-lyrics#note-4307282\&quot; data-id=\&quot;4307282\&quot; class=\&quot;referent\&quot; ng-click=\&quot;open()\&quot; ng-class=\&quot;{\n          'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),\n          'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),\n          'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)\n        }\&quot; prevent-default-click=\&quot;\&quot; annotation-fragment=\&quot;4307282\&quot; on-hover-with-no-digest=\&quot;set_current_hover_and_digest(hover ? fragment_id : undefined)\&quot; classification=\&quot;accepted\&quot; image=\&quot;false\&quot; pending-editorial-actions-count=\&quot;0\&quot;&gt;I got a cute face, chubby waist&lt;/a&gt;&lt;br&gt;\nThick legs, in shape&lt;br&gt;\nRump shakin, both ways&lt;br&gt;\nMake you do a double take&lt;br&gt;\n&lt;a href=\&quot;/Missy-elliott-lose-control-lyrics#note-1760219\&quot; data-id=\&quot;1760219\&quot; class=\&quot;referent\&quot; ng-click=\&quot;open()\&quot; ng-class=\&quot;{\n          'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),\n          'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),\n          'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)\n        }\&quot; prevent-default-click=\&quot;\&quot; annotation-fragment=\&quot;1760219\&quot; on-hover-with-no-digest=\&quot;set_current_hover_and_digest(hover ? fragment_id : undefined)\&quot; classification=\&quot;accepted\&quot; image=\&quot;false\&quot; pending-editorial-actions-count=\&quot;0\&quot;&gt;Planet rocker,&lt;/a&gt; show stopper&lt;br&gt;\nFlow proper, head knocker&lt;br&gt;\nBeat scholar, tail dropper&lt;br&gt;\nDo my thang, motherfucker&lt;br&gt;\nMy Rolls Royce, Lamborghini&lt;br&gt;\nBlue Madena, always beamin&lt;br&gt;\nRag top, chrome pipes&lt;br&gt;\nBlue lights, outta sight&lt;br&gt;\n(Long weave) sewed in&lt;br&gt;\n(Say it again) sewed in&lt;br&gt;\nMake that money, throw it in&lt;br&gt;\nBooty bouncin, gone head&lt;br&gt;\n&lt;br&gt;\n[Hook: Ciara &amp;amp; Missy Elliot]&lt;br&gt;\nEverybody here - get it out of control&lt;br&gt;\nGet your backs off the wall, cause &lt;a href=\&quot;/Missy-elliott-lose-control-lyrics#note-3480729\&quot; data-id=\&quot;3480729\&quot; class=\&quot;referent\&quot; ng-click=\&quot;open()\&quot; ng-class=\&quot;{\n          'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),\n          'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),\n          'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)\n        }\&quot; prevent-default-click=\&quot;\&quot; annotation-fragment=\&quot;3480729\&quot; on-hover-with-no-digest=\&quot;set_current_hover_and_digest(hover ? fragment_id : undefined)\&quot; classification=\&quot;unreviewed\&quot; image=\&quot;false\&quot; pending-editorial-actions-count=\&quot;0\&quot;&gt;Misdemeanor&lt;/a&gt; said so&lt;br&gt;\nEverybody, everybody, everybody, everybody&lt;br&gt;\n(Just throw your hands in the air!)&lt;br&gt;\n&lt;br&gt;\n[Verse 2: Ciara &amp;amp; (Missy Elliot)]&lt;br&gt;\nWell my name is Ciara, for all you fly fellas&lt;br&gt;\nNo one, can do it better (she'll sing on acapella)&lt;br&gt;\nBoy the music makes me lose control&lt;br&gt;\n&lt;br&gt;\n[Verse 3: Missy]&lt;br&gt;\n(Now bring it back now!) We gon' make you lose control&lt;br&gt;\nAnd let it go, 'fore you know, you gon' hit the flo'&lt;br&gt;\nI rock to the beat til I'm (tired)&lt;br&gt;\nI walk in the club it's (fire)&lt;br&gt;\nGet it crunk and wired&lt;br&gt;\nWave your hands scream (louder)&lt;br&gt;\nIf you smoke then fire it up&lt;br&gt;\nBrang the roof down and (holla)&lt;br&gt;\nIf you tipsy stand up&lt;br&gt;\nDJ turn it (louder)&lt;br&gt;\nTake somebody by the waist and (uhh!)&lt;br&gt;\nNow throw it in they face like (uhh!)&lt;br&gt;\nHypnotic robotic, this here will rock yo' bodies&lt;br&gt;\nTake somebody by the waist and (uhh!)&lt;br&gt;\nNow throw it in they face like (uhh!)&lt;br&gt;\nSystematic ecstatic (THIS HIT BE AUTOMATIC)&lt;br&gt;\n&lt;br&gt;\n[Bridge - Missy &amp;amp; Fatman Scoop]&lt;br&gt;\nWork me, work, work&lt;br&gt;\nWork me, work, work&lt;br&gt;\nWork me, work, work&lt;br&gt;\nWork me, do it right&lt;br&gt;\nHit the floor, hit the floor&lt;br&gt;\nHit the floor, hit the floor&lt;br&gt;\nHit the floor, hit the floor&lt;br&gt;\nHit the floor, hit the floor&lt;br&gt;\n&lt;br&gt;\n[Hook - Missy &amp;amp; Fatman]&lt;br&gt;\nEverybody here - get it out of control&lt;br&gt;\nGet your backs off the wall, cause Misdemeanor said so&lt;br&gt;\nEverybody, everybody, everybody, everybody&lt;br&gt;\n(Just throw your hands in the air!)&lt;br&gt;\nEverybody here - get it out of control&lt;br&gt;\nGet your backs off the wall, cause Misdemeanor said so&lt;br&gt;\nEverybody, everybody, everybody, everybody&lt;br&gt;\n(Just throw your hands in the air!)&lt;br&gt;\n&lt;br&gt;\n[Fatman Scoop] + (Missy)&lt;br&gt;\nGet your back off the wall, get your back off the wall&lt;br&gt;\nGet your back off the wall, get your back off the wall&lt;br&gt;\n(Everybody, get loose) Now put your back, on the wall&lt;br&gt;\nPut your back, on the wall&lt;br&gt;\nPut your back, on the wall, put your back, on the wall&lt;br&gt;\nMisdemeanor's in the house&lt;br&gt;\nYeah, Ciara's in the house&lt;br&gt;\nMisdemeanor's in the house, \&quot;Music make you lose control\&quot;&lt;br&gt;\nWe on fire, we on fire, we on fire, we on fire&lt;br&gt;\nNow throw it girl, throw it girl, throw it girl, yes&lt;br&gt;\nNow move your arms to the left girl&lt;br&gt;\nNow move your arms to the left girl&lt;br&gt;\nNow move your arms to the right girl&lt;br&gt;\nNow move your arms to the right girl&lt;br&gt;\nLet's go now, let's go now, let's go now, WOO! Let's go&lt;br&gt;\nShould I bring it back right now?&lt;br&gt;\nNow bring it back down!&lt;br&gt;\nWOO! Oh, I see you &lt;a href=\&quot;/Missy-elliott-lose-control-lyrics#note-15871347\&quot; data-id=\&quot;15871347\&quot; class=\&quot;referent\&quot; ng-click=\&quot;open()\&quot; ng-class=\&quot;{\n          'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),\n          'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),\n          'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)\n        }\&quot; prevent-default-click=\&quot;\&quot; annotation-fragment=\&quot;15871347\&quot; on-hover-with-no-digest=\&quot;set_current_hover_and_digest(hover ? fragment_id : undefined)\&quot; classification=\&quot;unreviewed\&quot; image=\&quot;false\&quot; pending-editorial-actions-count=\&quot;0\&quot;&gt;C&lt;/a&gt;&lt;br&gt;\nNow see, I'mma I'mma do it like C do it&lt;br&gt;\nNow shake it girl, c'mon and just shake it girl&lt;br&gt;\nC'mon and let it pop right girl, c'mon and let it pop right girl&lt;br&gt;\nNow, now, now back it up girl, back it up girl&lt;br&gt;\nBack it up girl, back it up girl&lt;br&gt;\nWOO! WOO! WOO! Yo, yo&lt;br&gt;\nBring it to the front girl, yo, yo&lt;br&gt;\nBring it to the front girl, yo, yo&lt;br&gt;\nBring it to the front girl, yo, yo&lt;br&gt;\nBring it to the front girl, let's go, let's go&lt;/p&gt;\n\n&quot;}},&quot;next_track&quot;:{&quot;number&quot;:5,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;annotation_count&quot;:1,&quot;api_path&quot;:&quot;/songs/33164&quot;,&quot;full_title&quot;:&quot;My Struggles by Missy Elliott (Ft. Grand Puba &amp; Mary J. Blige)&quot;,&quot;header_image_thumbnail_url&quot;:&quot;https://images.genius.com/49c23ba7107cf9bb93749df01b30111c.300x300x1.jpg&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/49c23ba7107cf9bb93749df01b30111c.1000x1000x1.jpg&quot;,&quot;id&quot;:33164,&quot;instrumental&quot;:false,&quot;lyrics_owner_id&quot;:50,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;lyrics_updated_at&quot;:1501007805,&quot;path&quot;:&quot;/Missy-elliott-my-struggles-lyrics&quot;,&quot;pyongs_count&quot;:null,&quot;song_art_image_thumbnail_url&quot;:&quot;https://images.genius.com/49c23ba7107cf9bb93749df01b30111c.300x300x1.jpg&quot;,&quot;stats&quot;:{&quot;hot&quot;:false,&quot;unreviewed_annotations&quot;:1},&quot;title&quot;:&quot;My Struggles&quot;,&quot;title_with_featured&quot;:&quot;My Struggles (Ft. Grand Puba &amp; Mary J. Blige)&quot;,&quot;updated_by_human_at&quot;:1539978920,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-my-struggles-lyrics&quot;,&quot;primary_artist&quot;:{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1529&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/fc183acc7f3e70189a76afe4a0d8cdf7.240x320x1.jpg&quot;,&quot;id&quot;:1529,&quot;image_url&quot;:&quot;https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg&quot;,&quot;index_character&quot;:&quot;m&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Missy Elliott&quot;,&quot;slug&quot;:&quot;Missy-elliott&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Missy-elliott&quot;}}},&quot;pinned_questions&quot;:[],&quot;preloaded_referents&quot;:[{&quot;_type&quot;:&quot;referent&quot;,&quot;annotator_id&quot;:129657,&quot;annotator_login&quot;:&quot;LowKey123&quot;,&quot;api_path&quot;:&quot;/referents/3480729&quot;,&quot;classification&quot;:&quot;unreviewed&quot;,&quot;fragment&quot;:&quot;Misdemeanor&quot;,&quot;id&quot;:3480729,&quot;ios_app_url&quot;:&quot;genius://referents/3480729&quot;,&quot;is_description&quot;:false,&quot;is_image&quot;:false,&quot;path&quot;:&quot;/3480729/Missy-elliott-lose-control/Misdemeanor&quot;,&quot;range&quot;:{&quot;content&quot;:&quot;Misdemeanor&quot;},&quot;song_id&quot;:33158,&quot;url&quot;:&quot;https://genius.com/3480729/Missy-elliott-lose-control/Misdemeanor&quot;,&quot;verified_annotator_ids&quot;:[],&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;add_pinned_annotation_to&quot;,&quot;add_community_annotation_to&quot;],&quot;relationships&quot;:{}},&quot;tracking_paths&quot;:{&quot;aggregate&quot;:&quot;/3480729/Missy-elliott-lose-control/Misdemeanor&quot;,&quot;concurrent&quot;:&quot;/Missy-elliott-lose-control-lyrics&quot;},&quot;twitter_share_message&quot;:&quot;“One of Missy Elliots nicknames.” —@Genius&quot;,&quot;annotatable&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33158&quot;,&quot;client_timestamps&quot;:{&quot;updated_by_human_at&quot;:1543136767,&quot;lyrics_updated_at&quot;:1543136767},&quot;context&quot;:&quot;Missy Elliott&quot;,&quot;id&quot;:33158,&quot;image_url&quot;:&quot;https://images.genius.com/51639e476eac5cb5bed5794c023763ff.700x707x1.jpg&quot;,&quot;link_title&quot;:&quot;Lose Control by Missy Elliott (Ft. Ciara &amp; Fatman Scoop)&quot;,&quot;title&quot;:&quot;Lose Control&quot;,&quot;type&quot;:&quot;Song&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-lose-control-lyrics&quot;},&quot;annotations&quot;:[{&quot;_type&quot;:&quot;annotation&quot;,&quot;api_path&quot;:&quot;/annotations/3480729&quot;,&quot;being_created&quot;:false,&quot;body&quot;:{&quot;html&quot;:&quot;&lt;p&gt;One of Missy Elliots nicknames.&lt;/p&gt;\n\n&lt;p&gt;&lt;img src=\&quot;https://images.rapgenius.com/20cd640901fc43bf0560824e9ee3544f.500x447x1.jpg\&quot; alt=\&quot;\&quot; width=\&quot;500\&quot; height=\&quot;447\&quot; data-animated=\&quot;false\&quot;&gt;&lt;/p&gt;&quot;,&quot;plain&quot;:&quot;One of Missy Elliots nicknames.&quot;,&quot;markdown&quot;:&quot;One of Missy Elliots nicknames.http://images.rapgenius.com/20cd640901fc43bf0560824e9ee3544f.500x447x1.jpg&quot;},&quot;comment_count&quot;:0,&quot;community&quot;:true,&quot;created_at&quot;:1406555725,&quot;custom_preview&quot;:null,&quot;deleted&quot;:false,&quot;embed_content&quot;:&quot;&lt;blockquote class='rg_standalone_container' data-src='//genius.com/annotations/3480729/standalone_embed'&gt;&lt;a href='https://genius.com/3480729/Missy-elliott-lose-control/Misdemeanor'&gt;Misdemeanor&lt;/a&gt;&lt;br&gt;&lt;a href='https://genius.com/Missy-elliott-lose-control-lyrics'&gt;&amp;#8213; Missy Elliott (Ft. Ciara &amp; Fatman Scoop) – Lose Control&lt;/a&gt;&lt;/blockquote&gt;&lt;script async crossorigin src='//genius.com/annotations/load_standalone_embeds.js'&gt;&lt;/script&gt;&quot;,&quot;has_voters&quot;:true,&quot;id&quot;:3480729,&quot;needs_exegesis&quot;:false,&quot;pinned&quot;:false,&quot;proposed_edit_count&quot;:0,&quot;pyongs_count&quot;:null,&quot;referent_id&quot;:3480729,&quot;share_url&quot;:&quot;https://genius.com/3480729&quot;,&quot;source&quot;:null,&quot;state&quot;:&quot;pending&quot;,&quot;twitter_share_message&quot;:&quot;“One of Missy Elliots nicknames.” —@Genius&quot;,&quot;url&quot;:&quot;https://genius.com/3480729/Missy-elliott-lose-control/Misdemeanor&quot;,&quot;verified&quot;:false,&quot;votes_total&quot;:4,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[&quot;create_comment&quot;],&quot;excluded_permissions&quot;:[&quot;vote&quot;,&quot;edit&quot;,&quot;cosign&quot;,&quot;uncosign&quot;,&quot;destroy&quot;,&quot;accept&quot;,&quot;reject&quot;,&quot;see_unreviewed&quot;,&quot;clear_votes&quot;,&quot;propose_edit_to&quot;,&quot;pin_to_profile&quot;,&quot;unpin_from_profile&quot;,&quot;update_source&quot;,&quot;edit_custom_preview&quot;],&quot;interactions&quot;:{&quot;cosign&quot;:false,&quot;pyong&quot;:false,&quot;vote&quot;:null},&quot;iq_by_action&quot;:{}},&quot;accepted_by&quot;:null,&quot;authors&quot;:[{&quot;_type&quot;:&quot;user_attribution&quot;,&quot;attribution&quot;:1.0,&quot;pinned_role&quot;:null,&quot;user&quot;:{&quot;_type&quot;:&quot;user&quot;,&quot;about_me_summary&quot;:&quot;Last Muthaf**ka Breathin'&quot;,&quot;api_path&quot;:&quot;/users/129657&quot;,&quot;avatar&quot;:{&quot;tiny&quot;:{&quot;url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/tiny/129657_me3.jpg&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:16,&quot;height&quot;:16}},&quot;thumb&quot;:{&quot;url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/thumb/1358290523_129657_me3.jpg&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:32,&quot;height&quot;:32}},&quot;small&quot;:{&quot;url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/small/1358290523_129657_me3.jpg&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:100,&quot;height&quot;:100}},&quot;medium&quot;:{&quot;url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/medium/1358290523_129657_me3.jpg&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:300,&quot;height&quot;:400}}},&quot;header_image_url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/medium/1358290523_129657_me3.jpg&quot;,&quot;human_readable_role_for_display&quot;:&quot;Contributor&quot;,&quot;id&quot;:129657,&quot;iq&quot;:790,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;login&quot;:&quot;LowKey123&quot;,&quot;name&quot;:&quot;LowKey123&quot;,&quot;role_for_display&quot;:&quot;contributor&quot;,&quot;url&quot;:&quot;https://genius.com/LowKey123&quot;,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;follow&quot;],&quot;interactions&quot;:{&quot;following&quot;:false},&quot;features&quot;:[]}}}],&quot;cosigned_by&quot;:[],&quot;created_by&quot;:{&quot;_type&quot;:&quot;user&quot;,&quot;about_me_summary&quot;:&quot;Last Muthaf**ka Breathin'&quot;,&quot;api_path&quot;:&quot;/users/129657&quot;,&quot;avatar&quot;:{&quot;tiny&quot;:{&quot;url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/tiny/129657_me3.jpg&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:16,&quot;height&quot;:16}},&quot;thumb&quot;:{&quot;url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/thumb/1358290523_129657_me3.jpg&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:32,&quot;height&quot;:32}},&quot;small&quot;:{&quot;url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/small/1358290523_129657_me3.jpg&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:100,&quot;height&quot;:100}},&quot;medium&quot;:{&quot;url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/medium/1358290523_129657_me3.jpg&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:300,&quot;height&quot;:400}}},&quot;header_image_url&quot;:&quot;https://s3.amazonaws.com/rapgenius/avatars/medium/1358290523_129657_me3.jpg&quot;,&quot;human_readable_role_for_display&quot;:&quot;Contributor&quot;,&quot;id&quot;:129657,&quot;iq&quot;:790,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;login&quot;:&quot;LowKey123&quot;,&quot;name&quot;:&quot;LowKey123&quot;,&quot;role_for_display&quot;:&quot;contributor&quot;,&quot;url&quot;:&quot;https://genius.com/LowKey123&quot;,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;follow&quot;],&quot;interactions&quot;:{&quot;following&quot;:false},&quot;features&quot;:[]}},&quot;rejection_comment&quot;:null,&quot;top_comment&quot;:null,&quot;verified_by&quot;:null}]}],&quot;primary_album_tracks&quot;:[{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:1,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33159&quot;,&quot;id&quot;:33159,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-joy-lyrics&quot;,&quot;title&quot;:&quot;Joy&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-joy-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:2,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33161&quot;,&quot;id&quot;:33161,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-partytime-lyrics&quot;,&quot;title&quot;:&quot;Partytime&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-partytime-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:3,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33155&quot;,&quot;id&quot;:33155,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-irresistible-delicious-lyrics&quot;,&quot;title&quot;:&quot;Irresistible Delicious&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-irresistible-delicious-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:4,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33158&quot;,&quot;id&quot;:33158,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-lose-control-lyrics&quot;,&quot;title&quot;:&quot;Lose Control&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-lose-control-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:5,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33164&quot;,&quot;id&quot;:33164,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-my-struggles-lyrics&quot;,&quot;title&quot;:&quot;My Struggles&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-my-struggles-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:6,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33156&quot;,&quot;id&quot;:33156,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-meltdown-lyrics&quot;,&quot;title&quot;:&quot;Meltdown&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-meltdown-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:7,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33160&quot;,&quot;id&quot;:33160,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-on-and-on-lyrics&quot;,&quot;title&quot;:&quot;On &amp; On&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-on-and-on-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:8,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33162&quot;,&quot;id&quot;:33162,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-we-run-this-lyrics&quot;,&quot;title&quot;:&quot;We Run This&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-we-run-this-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:9,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33163&quot;,&quot;id&quot;:33163,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-remember-when-lyrics&quot;,&quot;title&quot;:&quot;Remember When&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-remember-when-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:10,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33152&quot;,&quot;id&quot;:33152,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-4-my-man-lyrics&quot;,&quot;title&quot;:&quot;4 My Man&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-4-my-man-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:11,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33157&quot;,&quot;id&quot;:33157,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-cant-stop-lyrics&quot;,&quot;title&quot;:&quot;Can't Stop&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-cant-stop-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:12,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33165&quot;,&quot;id&quot;:33165,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-teary-eyed-lyrics&quot;,&quot;title&quot;:&quot;Teary Eyed&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-teary-eyed-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:13,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33172&quot;,&quot;id&quot;:33172,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-mommy-lyrics&quot;,&quot;title&quot;:&quot;Mommy&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-mommy-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:14,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33154&quot;,&quot;id&quot;:33154,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-click-clack-lyrics&quot;,&quot;title&quot;:&quot;Click Clack&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-click-clack-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:15,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33166&quot;,&quot;id&quot;:33166,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-time-and-time-again-lyrics&quot;,&quot;title&quot;:&quot;Time and Time Again&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-time-and-time-again-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:16,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33151&quot;,&quot;id&quot;:33151,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-bad-man-lyrics&quot;,&quot;title&quot;:&quot;Bad Man&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-bad-man-lyrics&quot;}},{&quot;_type&quot;:&quot;album_appearance&quot;,&quot;number&quot;:null,&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/3079936&quot;,&quot;id&quot;:3079936,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;path&quot;:&quot;/Missy-elliott-the-cookbook-tracklist-lyrics&quot;,&quot;title&quot;:&quot;The Cookbook [Tracklist]&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-the-cookbook-tracklist-lyrics&quot;}}],&quot;recirculated_content&quot;:[{&quot;metadata&quot;:{&quot;target&quot;:&quot;article&quot;,&quot;type&quot;:&quot;news&quot;,&quot;id&quot;:5517},&quot;appearance&quot;:{&quot;attribution&quot;:&quot;Grant Rindner&quot;,&quot;image_url&quot;:&quot;https://images.genius.com/759f07f6e0afebd791e5463aa93e1801.1800x1000x1.png&quot;,&quot;label&quot;:&quot;news&quot;,&quot;overlay_compatible_image_url&quot;:&quot;https://images.genius.com/759f07f6e0afebd791e5463aa93e1801.1800x1000x1.png&quot;,&quot;publish_date&quot;:1532807520,&quot;title&quot;:&quot;Ciara And Missy Elliott Team Up For First Time In Nearly A Decade On “Level Up” Remix&quot;},&quot;behavior&quot;:{&quot;attributes&quot;:{&quot;api_path&quot;:&quot;/articles/5517&quot;,&quot;url&quot;:&quot;https://genius.com/a/ciara-and-missy-elliott-team-up-for-first-time-in-nearly-a-decade-on-level-up-remix&quot;},&quot;classification&quot;:&quot;link&quot;,&quot;video&quot;:null}}],&quot;song&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;annotation_count&quot;:7,&quot;api_path&quot;:&quot;/songs/33158&quot;,&quot;apple_music_id&quot;:&quot;73240226&quot;,&quot;apple_music_player_url&quot;:&quot;https://genius.com/songs/33158/apple_music_player&quot;,&quot;comment_count&quot;:1,&quot;custom_header_image_url&quot;:null,&quot;custom_song_art_image_url&quot;:&quot;http://images.genius.com/51639e476eac5cb5bed5794c023763ff.700x707x1.jpg&quot;,&quot;description&quot;:{&quot;html&quot;:&quot;&lt;p&gt;Another classic Missy Elliott banger, centred around the loss of control of your extremities on the dance floor. It samples Hot Streak’s &lt;a href=\&quot;https://www.youtube.com/watch?v=HQiL-iHsVu0\&quot; rel=\&quot;noopener nofollow\&quot;&gt;“Body Work”&lt;/a&gt; and Cybotron’s &lt;a href=\&quot;https://genius.com/Cybotron-clear-lyrics\&quot; rel=\&quot;noopener\&quot; data-api_path=\&quot;/songs/1977813\&quot;&gt;“Clear,”&lt;/a&gt; and produced by Missy herself, featuring Ciara, off the strength of her incredible &lt;a href=\&quot;https://genius.com/albums/Ciara/Goodies\&quot; rel=\&quot;noopener\&quot; data-api_path=\&quot;/albums/1561\&quot;&gt;&lt;em&gt;Goodies&lt;/em&gt;&lt;/a&gt; album, and Fatman Scoop as hype man.&lt;/p&gt;\n\n&lt;p&gt;The &lt;a href=\&quot;https://www.youtube.com/watch?v=khgIVMUvihg\&quot; rel=\&quot;noopener nofollow\&quot;&gt;video&lt;/a&gt; features a bunch of dancers engaging in futuristic moves in various settings, with Missy displaying &lt;a href=\&quot;https://youtu.be/khgIVMUvihg?t=1m2s\&quot; rel=\&quot;noopener nofollow\&quot;&gt;cutting edge technology.&lt;/a&gt;&lt;/p&gt;\n\n&lt;p&gt;The song was a chart smash, making it to number 3 on the &lt;em&gt;US Billboard&lt;/em&gt; charts, and top 10 in 4 other countries.&lt;/p&gt;&quot;,&quot;markdown&quot;:&quot;Another classic Missy Elliott banger, centred around the loss of control of your extremities on the dance floor. It samples Hot Streak's [\&quot;Body Work\&quot;](https://www.youtube.com/watch?v=HQiL-iHsVu0) and Cybotron's [\&quot;Clear,\&quot;](http://genius.com/Cybotron-clear-lyrics) and produced by Missy herself, featuring Ciara, off the strength of her incredible [*Goodies*](http://genius.com/albums/Ciara/Goodies) album, and Fatman Scoop as hype man.\n\nThe [video](https://www.youtube.com/watch?v=khgIVMUvihg) features a bunch of dancers engaging in futuristic moves in various settings, with Missy displaying [cutting edge technology.](https://youtu.be/khgIVMUvihg?t=1m2s) \n\nThe song was a chart smash, making it to number 3 on the *US Billboard* charts, and top 10 in 4 other countries.&quot;},&quot;description_preview&quot;:&quot;Another classic Missy Elliott banger, centred around the loss of control of your extremities on the dance floor. It samples Hot Streak’s “Body Work” and Cybotron’s “Clear,” and produced by Missy herself, featuring Ciara, off the strength of her incredible Goodies album, and Fatman Scoop as hype man.\n\nThe video features a bunch of dancers engaging in futuristic moves in various settings, with Missy displaying cutting edge technology.\n\nThe song was a chart smash, making it to number 3 on the US Billboard charts, and top 10 in 4 other countries.&quot;,&quot;embed_content&quot;:&quot;&lt;div id='rg_embed_link_33158' class='rg_embed_link' data-song-id='33158'&gt;Read &lt;a href='https://genius.com/Missy-elliott-lose-control-lyrics'&gt;“Lose Control” by Missy Elliott&lt;/a&gt; on Genius&lt;/div&gt; &lt;script crossorigin src='//genius.com/songs/33158/embed.js'&gt;&lt;/script&gt;&quot;,&quot;featured_video&quot;:true,&quot;full_title&quot;:&quot;Lose Control by Missy Elliott (Ft. Ciara &amp; Fatman Scoop)&quot;,&quot;header_image_thumbnail_url&quot;:&quot;https://images.genius.com/51639e476eac5cb5bed5794c023763ff.300x303x1.jpg&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/51639e476eac5cb5bed5794c023763ff.700x707x1.jpg&quot;,&quot;hidden&quot;:false,&quot;id&quot;:33158,&quot;instrumental&quot;:false,&quot;is_music&quot;:true,&quot;lyrics_owner_id&quot;:50,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;lyrics_updated_at&quot;:1543136767,&quot;path&quot;:&quot;/Missy-elliott-lose-control-lyrics&quot;,&quot;pending_lyrics_edits_count&quot;:0,&quot;published&quot;:false,&quot;pusher_channel&quot;:&quot;song-33158&quot;,&quot;pyongs_count&quot;:8,&quot;recording_location&quot;:null,&quot;release_date&quot;:&quot;2005-05-27&quot;,&quot;release_date_components&quot;:{&quot;year&quot;:2005,&quot;month&quot;:5,&quot;day&quot;:27},&quot;share_url&quot;:&quot;https://genius.com/Missy-elliott-lose-control-lyrics&quot;,&quot;song_art_image_thumbnail_url&quot;:&quot;https://images.genius.com/51639e476eac5cb5bed5794c023763ff.300x303x1.jpg&quot;,&quot;song_art_image_url&quot;:&quot;https://images.genius.com/51639e476eac5cb5bed5794c023763ff.700x707x1.jpg&quot;,&quot;soundcloud_url&quot;:null,&quot;spotify_uuid&quot;:&quot;0UaMYEvWZi0ZqiDOoHU3YI&quot;,&quot;stats&quot;:{&quot;accepted_annotations&quot;:4,&quot;contributors&quot;:20,&quot;hot&quot;:false,&quot;iq_earners&quot;:20,&quot;transcribers&quot;:0,&quot;unreviewed_annotations&quot;:2,&quot;verified_annotations&quot;:0,&quot;concurrents&quot;:2,&quot;pageviews&quot;:55627},&quot;title&quot;:&quot;Lose Control&quot;,&quot;title_with_featured&quot;:&quot;Lose Control (Ft. Ciara &amp; Fatman Scoop)&quot;,&quot;tracking_data&quot;:[{&quot;key&quot;:&quot;Song ID&quot;,&quot;value&quot;:33158},{&quot;key&quot;:&quot;Title&quot;,&quot;value&quot;:&quot;Lose Control&quot;},{&quot;key&quot;:&quot;Primary Artist&quot;,&quot;value&quot;:&quot;Missy Elliott&quot;},{&quot;key&quot;:&quot;Primary Artist ID&quot;,&quot;value&quot;:1529},{&quot;key&quot;:&quot;Primary Album&quot;,&quot;value&quot;:&quot;The Cookbook&quot;},{&quot;key&quot;:&quot;Primary Album ID&quot;,&quot;value&quot;:6618},{&quot;key&quot;:&quot;Tag&quot;,&quot;value&quot;:&quot;rap&quot;},{&quot;key&quot;:&quot;Primary Tag&quot;,&quot;value&quot;:&quot;rap&quot;},{&quot;key&quot;:&quot;Primary Tag ID&quot;,&quot;value&quot;:1434},{&quot;key&quot;:&quot;Music?&quot;,&quot;value&quot;:true},{&quot;key&quot;:&quot;Annotatable Type&quot;,&quot;value&quot;:&quot;Song&quot;},{&quot;key&quot;:&quot;Annotatable ID&quot;,&quot;value&quot;:33158},{&quot;key&quot;:&quot;featured_video&quot;,&quot;value&quot;:true},{&quot;key&quot;:&quot;cohort_ids&quot;,&quot;value&quot;:[]},{&quot;key&quot;:&quot;has_verified_callout&quot;,&quot;value&quot;:false},{&quot;key&quot;:&quot;has_featured_annotation&quot;,&quot;value&quot;:true},{&quot;key&quot;:&quot;created_at&quot;,&quot;value&quot;:&quot;2011-04-09T06:23:13Z&quot;},{&quot;key&quot;:&quot;created_month&quot;,&quot;value&quot;:&quot;2011-04-01&quot;},{&quot;key&quot;:&quot;created_year&quot;,&quot;value&quot;:2011},{&quot;key&quot;:&quot;song_tier&quot;,&quot;value&quot;:&quot;D&quot;},{&quot;key&quot;:&quot;Has Recirculated Articles&quot;,&quot;value&quot;:true},{&quot;key&quot;:&quot;Lyrics Language&quot;,&quot;value&quot;:&quot;en&quot;},{&quot;key&quot;:&quot;Has Song Story&quot;,&quot;value&quot;:false},{&quot;key&quot;:&quot;Song Story ID&quot;,&quot;value&quot;:null},{&quot;key&quot;:&quot;Has Apple Match&quot;,&quot;value&quot;:true}],&quot;tracking_paths&quot;:{&quot;aggregate&quot;:&quot;/Missy-elliott-lose-control-lyrics&quot;,&quot;concurrent&quot;:&quot;/Missy-elliott-lose-control-lyrics&quot;},&quot;twitter_share_message&quot;:&quot;Missy Elliott – Lose Control @MissyElliott https://genius.com/Missy-elliott-lose-control-lyrics&quot;,&quot;twitter_share_message_without_url&quot;:&quot;Missy Elliott – Lose Control @MissyElliott&quot;,&quot;updated_by_human_at&quot;:1543136767,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-lose-control-lyrics&quot;,&quot;viewable_by_roles&quot;:[],&quot;vttp_id&quot;:null,&quot;youtube_start&quot;:&quot;0&quot;,&quot;youtube_url&quot;:&quot;http://www.youtube.com/watch?v=HrE5lrjBryI&quot;,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[&quot;see_pageviews&quot;,&quot;view_apple_music_player&quot;,&quot;create_comment&quot;,&quot;view_song_story_gallery&quot;],&quot;excluded_permissions&quot;:[&quot;follow&quot;,&quot;award_transcription_iq&quot;,&quot;remove_transcription_iq&quot;,&quot;pyong&quot;,&quot;edit_lyrics&quot;,&quot;view_annotation_engagement_data&quot;,&quot;publish&quot;,&quot;unpublish&quot;,&quot;edit_spotify_details&quot;,&quot;hide&quot;,&quot;unhide&quot;,&quot;toggle_featured_video&quot;,&quot;add_pinned_annotation_to&quot;,&quot;add_community_annotation_to&quot;,&quot;destroy&quot;,&quot;mark_as_not_spam&quot;,&quot;edit_spotify_annotations_for&quot;,&quot;verify_lyrics&quot;,&quot;unverify_lyrics&quot;,&quot;edit_anything&quot;,&quot;edit_album_appearances&quot;,&quot;edit_any_media&quot;,&quot;edit&quot;,&quot;rename&quot;,&quot;edit_tags&quot;,&quot;watch_fact_track&quot;,&quot;reindex&quot;,&quot;view_lyrics_synchronization&quot;,&quot;enable_media&quot;,&quot;disable_media&quot;,&quot;edit_lyrics_or_annotation_brackets&quot;,&quot;see_editorial_indicators&quot;,&quot;view_attribution_visualization&quot;,&quot;edit_annotation_brackets&quot;,&quot;preview_lyrics_for_export&quot;,&quot;hide_apple_player&quot;,&quot;unhide_apple_player&quot;,&quot;trigger_apple_match&quot;,&quot;mark_lyrics_evaluation_as_approved&quot;,&quot;mark_lyrics_evaluation_as_staff_approved&quot;,&quot;mark_lyrics_evaluation_as_unapproved&quot;,&quot;mark_lyrics_evaluation_as_un_staff_approved&quot;,&quot;view_transcriber_media_player&quot;,&quot;override_apple_match&quot;,&quot;edit_youtube_url&quot;,&quot;edit_soundcloud_url&quot;,&quot;edit_spotify_uuid&quot;,&quot;edit_vevo_url&quot;,&quot;moderate_annotations&quot;,&quot;create_annotation&quot;,&quot;see_short_id&quot;,&quot;manage_chart_item&quot;,&quot;create_tag&quot;,&quot;propose_lyrics_edit&quot;,&quot;moderate_lyrics_edit_proposals&quot;],&quot;interactions&quot;:{&quot;pyong&quot;:false,&quot;following&quot;:false},&quot;relationships&quot;:{},&quot;iq_by_action&quot;:{}},&quot;album&quot;:{&quot;_type&quot;:&quot;album&quot;,&quot;api_path&quot;:&quot;/albums/6618&quot;,&quot;cover_art_thumbnail_url&quot;:&quot;https://images.genius.com/49c23ba7107cf9bb93749df01b30111c.300x300x1.jpg&quot;,&quot;cover_art_url&quot;:&quot;https://images.genius.com/49c23ba7107cf9bb93749df01b30111c.1000x1000x1.jpg&quot;,&quot;full_title&quot;:&quot;The Cookbook by Missy Elliott&quot;,&quot;id&quot;:6618,&quot;name&quot;:&quot;The Cookbook&quot;,&quot;name_with_artist&quot;:&quot;The Cookbook (artist: Missy Elliott)&quot;,&quot;release_date_components&quot;:{&quot;year&quot;:2005,&quot;month&quot;:7,&quot;day&quot;:4},&quot;url&quot;:&quot;https://genius.com/albums/Missy-elliott/The-cookbook&quot;,&quot;artist&quot;:{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1529&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/fc183acc7f3e70189a76afe4a0d8cdf7.240x320x1.jpg&quot;,&quot;id&quot;:1529,&quot;image_url&quot;:&quot;https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg&quot;,&quot;index_character&quot;:&quot;m&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Missy Elliott&quot;,&quot;slug&quot;:&quot;Missy-elliott&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Missy-elliott&quot;}},&quot;albums&quot;:[{&quot;_type&quot;:&quot;album&quot;,&quot;api_path&quot;:&quot;/albums/6618&quot;,&quot;cover_art_thumbnail_url&quot;:&quot;https://images.genius.com/49c23ba7107cf9bb93749df01b30111c.300x300x1.jpg&quot;,&quot;cover_art_url&quot;:&quot;https://images.genius.com/49c23ba7107cf9bb93749df01b30111c.1000x1000x1.jpg&quot;,&quot;full_title&quot;:&quot;The Cookbook by Missy Elliott&quot;,&quot;id&quot;:6618,&quot;name&quot;:&quot;The Cookbook&quot;,&quot;name_with_artist&quot;:&quot;The Cookbook (artist: Missy Elliott)&quot;,&quot;release_date_components&quot;:{&quot;year&quot;:2005,&quot;month&quot;:7,&quot;day&quot;:4},&quot;url&quot;:&quot;https://genius.com/albums/Missy-elliott/The-cookbook&quot;,&quot;artist&quot;:{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1529&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/fc183acc7f3e70189a76afe4a0d8cdf7.240x320x1.jpg&quot;,&quot;id&quot;:1529,&quot;image_url&quot;:&quot;https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg&quot;,&quot;index_character&quot;:&quot;m&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Missy Elliott&quot;,&quot;slug&quot;:&quot;Missy-elliott&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Missy-elliott&quot;}},{&quot;_type&quot;:&quot;album&quot;,&quot;api_path&quot;:&quot;/albums/335070&quot;,&quot;cover_art_thumbnail_url&quot;:&quot;https://images.genius.com/75cfb85ca9e5a85370891a511a97b3d5.300x300x1.jpg&quot;,&quot;cover_art_url&quot;:&quot;https://images.genius.com/75cfb85ca9e5a85370891a511a97b3d5.640x640x1.jpg&quot;,&quot;full_title&quot;:&quot;Respect M.E. by Missy Elliott&quot;,&quot;id&quot;:335070,&quot;name&quot;:&quot;Respect M.E.&quot;,&quot;name_with_artist&quot;:&quot;Respect M.E. (artist: Missy Elliott)&quot;,&quot;release_date_components&quot;:{&quot;year&quot;:2006,&quot;month&quot;:9,&quot;day&quot;:4},&quot;url&quot;:&quot;https://genius.com/albums/Missy-elliott/Respect-m-e&quot;,&quot;artist&quot;:{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1529&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/fc183acc7f3e70189a76afe4a0d8cdf7.240x320x1.jpg&quot;,&quot;id&quot;:1529,&quot;image_url&quot;:&quot;https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg&quot;,&quot;index_character&quot;:&quot;m&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Missy Elliott&quot;,&quot;slug&quot;:&quot;Missy-elliott&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Missy-elliott&quot;}}],&quot;custom_performances&quot;:[{&quot;label&quot;:&quot;Label&quot;,&quot;artists&quot;:[{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/68978&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/365d2342d8e838709b3b0932e088a510.1000x667x1.jpg&quot;,&quot;id&quot;:68978,&quot;image_url&quot;:&quot;https://images.genius.com/204458cc33bf320629866a47bab1e342.1000x1000x1.png&quot;,&quot;index_character&quot;:&quot;a&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Atlantic Records&quot;,&quot;slug&quot;:&quot;Atlantic-records&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Atlantic-records&quot;}]}],&quot;description_annotation&quot;:{&quot;_type&quot;:&quot;referent&quot;,&quot;annotator_id&quot;:1299009,&quot;annotator_login&quot;:&quot;Theonlydjorkaeff&quot;,&quot;api_path&quot;:&quot;/referents/3523230&quot;,&quot;classification&quot;:&quot;accepted&quot;,&quot;fragment&quot;:&quot;Lose Control&quot;,&quot;id&quot;:3523230,&quot;ios_app_url&quot;:&quot;genius://referents/3523230&quot;,&quot;is_description&quot;:true,&quot;is_image&quot;:false,&quot;path&quot;:&quot;/3523230/Missy-elliott-lose-control/Lose-control&quot;,&quot;range&quot;:{&quot;content&quot;:&quot;Lose Control&quot;},&quot;song_id&quot;:33158,&quot;url&quot;:&quot;https://genius.com/3523230/Missy-elliott-lose-control/Lose-control&quot;,&quot;verified_annotator_ids&quot;:[],&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;add_pinned_annotation_to&quot;,&quot;add_community_annotation_to&quot;],&quot;relationships&quot;:{}},&quot;tracking_paths&quot;:{&quot;aggregate&quot;:&quot;/3523230/Missy-elliott-lose-control/Lose-control&quot;,&quot;concurrent&quot;:&quot;/Missy-elliott-lose-control-lyrics&quot;},&quot;twitter_share_message&quot;:&quot;“Another classic Missy Elliott banger, centred around the loss of control of your extremities on …” —@Genius&quot;,&quot;annotatable&quot;:{&quot;_type&quot;:&quot;song&quot;,&quot;api_path&quot;:&quot;/songs/33158&quot;,&quot;client_timestamps&quot;:{&quot;updated_by_human_at&quot;:1543136767,&quot;lyrics_updated_at&quot;:1543136767},&quot;context&quot;:&quot;Missy Elliott&quot;,&quot;id&quot;:33158,&quot;image_url&quot;:&quot;https://images.genius.com/51639e476eac5cb5bed5794c023763ff.700x707x1.jpg&quot;,&quot;link_title&quot;:&quot;Lose Control by Missy Elliott (Ft. Ciara &amp; Fatman Scoop)&quot;,&quot;title&quot;:&quot;Lose Control&quot;,&quot;type&quot;:&quot;Song&quot;,&quot;url&quot;:&quot;https://genius.com/Missy-elliott-lose-control-lyrics&quot;},&quot;annotations&quot;:[{&quot;_type&quot;:&quot;annotation&quot;,&quot;api_path&quot;:&quot;/annotations/3523230&quot;,&quot;being_created&quot;:false,&quot;body&quot;:{&quot;html&quot;:&quot;&lt;p&gt;Another classic Missy Elliott banger, centred around the loss of control of your extremities on the dance floor. It samples Hot Streak’s &lt;a href=\&quot;https://www.youtube.com/watch?v=HQiL-iHsVu0\&quot; rel=\&quot;noopener nofollow\&quot;&gt;“Body Work”&lt;/a&gt; and Cybotron’s &lt;a href=\&quot;https://genius.com/Cybotron-clear-lyrics\&quot; rel=\&quot;noopener\&quot; data-api_path=\&quot;/songs/1977813\&quot;&gt;“Clear,”&lt;/a&gt; and produced by Missy herself, featuring Ciara, off the strength of her incredible &lt;a href=\&quot;https://genius.com/albums/Ciara/Goodies\&quot; rel=\&quot;noopener\&quot; data-api_path=\&quot;/albums/1561\&quot;&gt;&lt;em&gt;Goodies&lt;/em&gt;&lt;/a&gt; album, and Fatman Scoop as hype man.&lt;/p&gt;\n\n&lt;p&gt;The &lt;a href=\&quot;https://www.youtube.com/watch?v=khgIVMUvihg\&quot; rel=\&quot;noopener nofollow\&quot;&gt;video&lt;/a&gt; features a bunch of dancers engaging in futuristic moves in various settings, with Missy displaying &lt;a href=\&quot;https://youtu.be/khgIVMUvihg?t=1m2s\&quot; rel=\&quot;noopener nofollow\&quot;&gt;cutting edge technology.&lt;/a&gt;&lt;/p&gt;\n\n&lt;p&gt;The song was a chart smash, making it to number 3 on the &lt;em&gt;US Billboard&lt;/em&gt; charts, and top 10 in 4 other countries.&lt;/p&gt;&quot;,&quot;markdown&quot;:&quot;Another classic Missy Elliott banger, centred around the loss of control of your extremities on the dance floor. It samples Hot Streak's [\&quot;Body Work\&quot;](https://www.youtube.com/watch?v=HQiL-iHsVu0) and Cybotron's [\&quot;Clear,\&quot;](http://genius.com/Cybotron-clear-lyrics) and produced by Missy herself, featuring Ciara, off the strength of her incredible [*Goodies*](http://genius.com/albums/Ciara/Goodies) album, and Fatman Scoop as hype man.\n\nThe [video](https://www.youtube.com/watch?v=khgIVMUvihg) features a bunch of dancers engaging in futuristic moves in various settings, with Missy displaying [cutting edge technology.](https://youtu.be/khgIVMUvihg?t=1m2s) \n\nThe song was a chart smash, making it to number 3 on the *US Billboard* charts, and top 10 in 4 other countries.&quot;},&quot;comment_count&quot;:0,&quot;community&quot;:true,&quot;created_at&quot;:1465289562,&quot;custom_preview&quot;:null,&quot;deleted&quot;:false,&quot;embed_content&quot;:&quot;&lt;blockquote class='rg_standalone_container' data-src='//genius.com/annotations/3523230/standalone_embed'&gt;&lt;a href='https://genius.com/3523230/Missy-elliott-lose-control/Lose-control'&gt;Lose Control&lt;/a&gt;&lt;br&gt;&lt;a href='https://genius.com/Missy-elliott-lose-control-lyrics'&gt;&amp;#8213; Missy Elliott (Ft. Ciara &amp; Fatman Scoop) – Lose Control&lt;/a&gt;&lt;/blockquote&gt;&lt;script async crossorigin src='//genius.com/annotations/load_standalone_embeds.js'&gt;&lt;/script&gt;&quot;,&quot;has_voters&quot;:true,&quot;id&quot;:3523230,&quot;needs_exegesis&quot;:false,&quot;pinned&quot;:false,&quot;proposed_edit_count&quot;:0,&quot;pyongs_count&quot;:2,&quot;referent_id&quot;:3523230,&quot;share_url&quot;:&quot;https://genius.com/3523230&quot;,&quot;source&quot;:null,&quot;state&quot;:&quot;accepted&quot;,&quot;twitter_share_message&quot;:&quot;“Another classic Missy Elliott banger, centred around the loss of control of your extremities on the danc…” —@Genius&quot;,&quot;url&quot;:&quot;https://genius.com/3523230/Missy-elliott-lose-control/Lose-control&quot;,&quot;verified&quot;:false,&quot;votes_total&quot;:4,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[&quot;create_comment&quot;],&quot;excluded_permissions&quot;:[&quot;vote&quot;,&quot;edit&quot;,&quot;cosign&quot;,&quot;uncosign&quot;,&quot;destroy&quot;,&quot;accept&quot;,&quot;reject&quot;,&quot;see_unreviewed&quot;,&quot;clear_votes&quot;,&quot;propose_edit_to&quot;,&quot;pin_to_profile&quot;,&quot;unpin_from_profile&quot;,&quot;update_source&quot;,&quot;edit_custom_preview&quot;],&quot;interactions&quot;:{&quot;cosign&quot;:false,&quot;pyong&quot;:false,&quot;vote&quot;:null},&quot;iq_by_action&quot;:{}},&quot;accepted_by&quot;:{&quot;_type&quot;:&quot;user&quot;,&quot;about_me_summary&quot;:&quot;Not around as much at the moment, if I don’t respond to your PM don’t take it personally!\n\nHush puppies?\n\nhttp://beardfood.com/\n\nhttp://www.bensbigblog.com.au/&quot;,&quot;api_path&quot;:&quot;/users/1299009&quot;,&quot;avatar&quot;:{&quot;tiny&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/tiny/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:16,&quot;height&quot;:16}},&quot;thumb&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/thumb/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:32,&quot;height&quot;:32}},&quot;small&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/small/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:100,&quot;height&quot;:100}},&quot;medium&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/medium/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:300,&quot;height&quot;:400}}},&quot;header_image_url&quot;:&quot;https://images.genius.com/2995f719dac028933921615f49aa7955.300x300x1.jpg&quot;,&quot;human_readable_role_for_display&quot;:&quot;Moderator&quot;,&quot;id&quot;:1299009,&quot;iq&quot;:687446,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;login&quot;:&quot;Theonlydjorkaeff&quot;,&quot;name&quot;:&quot;Ben Carter&quot;,&quot;role_for_display&quot;:&quot;moderator&quot;,&quot;url&quot;:&quot;https://genius.com/Theonlydjorkaeff&quot;,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;follow&quot;],&quot;interactions&quot;:{&quot;following&quot;:false},&quot;features&quot;:[]}},&quot;authors&quot;:[{&quot;_type&quot;:&quot;user_attribution&quot;,&quot;attribution&quot;:1.0,&quot;pinned_role&quot;:null,&quot;user&quot;:{&quot;_type&quot;:&quot;user&quot;,&quot;about_me_summary&quot;:&quot;Not around as much at the moment, if I don’t respond to your PM don’t take it personally!\n\nHush puppies?\n\nhttp://beardfood.com/\n\nhttp://www.bensbigblog.com.au/&quot;,&quot;api_path&quot;:&quot;/users/1299009&quot;,&quot;avatar&quot;:{&quot;tiny&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/tiny/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:16,&quot;height&quot;:16}},&quot;thumb&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/thumb/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:32,&quot;height&quot;:32}},&quot;small&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/small/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:100,&quot;height&quot;:100}},&quot;medium&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/medium/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:300,&quot;height&quot;:400}}},&quot;header_image_url&quot;:&quot;https://images.genius.com/2995f719dac028933921615f49aa7955.300x300x1.jpg&quot;,&quot;human_readable_role_for_display&quot;:&quot;Moderator&quot;,&quot;id&quot;:1299009,&quot;iq&quot;:687446,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;login&quot;:&quot;Theonlydjorkaeff&quot;,&quot;name&quot;:&quot;Ben Carter&quot;,&quot;role_for_display&quot;:&quot;moderator&quot;,&quot;url&quot;:&quot;https://genius.com/Theonlydjorkaeff&quot;,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;follow&quot;],&quot;interactions&quot;:{&quot;following&quot;:false},&quot;features&quot;:[]}}}],&quot;cosigned_by&quot;:[],&quot;created_by&quot;:{&quot;_type&quot;:&quot;user&quot;,&quot;about_me_summary&quot;:&quot;Not around as much at the moment, if I don’t respond to your PM don’t take it personally!\n\nHush puppies?\n\nhttp://beardfood.com/\n\nhttp://www.bensbigblog.com.au/&quot;,&quot;api_path&quot;:&quot;/users/1299009&quot;,&quot;avatar&quot;:{&quot;tiny&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/tiny/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:16,&quot;height&quot;:16}},&quot;thumb&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/thumb/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:32,&quot;height&quot;:32}},&quot;small&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/small/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:100,&quot;height&quot;:100}},&quot;medium&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/medium/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:300,&quot;height&quot;:400}}},&quot;header_image_url&quot;:&quot;https://images.genius.com/2995f719dac028933921615f49aa7955.300x300x1.jpg&quot;,&quot;human_readable_role_for_display&quot;:&quot;Moderator&quot;,&quot;id&quot;:1299009,&quot;iq&quot;:687446,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;login&quot;:&quot;Theonlydjorkaeff&quot;,&quot;name&quot;:&quot;Ben Carter&quot;,&quot;role_for_display&quot;:&quot;moderator&quot;,&quot;url&quot;:&quot;https://genius.com/Theonlydjorkaeff&quot;,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;follow&quot;],&quot;interactions&quot;:{&quot;following&quot;:false},&quot;features&quot;:[]}},&quot;rejection_comment&quot;:{&quot;_type&quot;:&quot;comment&quot;,&quot;api_path&quot;:&quot;/comments/3794562&quot;,&quot;body&quot;:{&quot;html&quot;:&quot;&quot;,&quot;markdown&quot;:&quot;&quot;},&quot;created_at&quot;:1465289236,&quot;has_voters&quot;:false,&quot;id&quot;:3794562,&quot;pinned_role&quot;:null,&quot;votes_total&quot;:0,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;vote&quot;,&quot;accept&quot;,&quot;reject&quot;,&quot;mark_spam&quot;,&quot;integrate&quot;,&quot;archive&quot;,&quot;destroy&quot;],&quot;interactions&quot;:{&quot;vote&quot;:null}},&quot;anonymous_author&quot;:null,&quot;author&quot;:{&quot;_type&quot;:&quot;user&quot;,&quot;about_me_summary&quot;:&quot;Not around as much at the moment, if I don’t respond to your PM don’t take it personally!\n\nHush puppies?\n\nhttp://beardfood.com/\n\nhttp://www.bensbigblog.com.au/&quot;,&quot;api_path&quot;:&quot;/users/1299009&quot;,&quot;avatar&quot;:{&quot;tiny&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/tiny/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:16,&quot;height&quot;:16}},&quot;thumb&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/thumb/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:32,&quot;height&quot;:32}},&quot;small&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/small/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:100,&quot;height&quot;:100}},&quot;medium&quot;:{&quot;url&quot;:&quot;https://images.genius.com/avatars/medium/1f81a10712fc9c886c8af720dc83c76f&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:300,&quot;height&quot;:400}}},&quot;header_image_url&quot;:&quot;https://images.genius.com/2995f719dac028933921615f49aa7955.300x300x1.jpg&quot;,&quot;human_readable_role_for_display&quot;:&quot;Moderator&quot;,&quot;id&quot;:1299009,&quot;iq&quot;:687446,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;login&quot;:&quot;Theonlydjorkaeff&quot;,&quot;name&quot;:&quot;Ben Carter&quot;,&quot;role_for_display&quot;:&quot;moderator&quot;,&quot;url&quot;:&quot;https://genius.com/Theonlydjorkaeff&quot;,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;follow&quot;],&quot;interactions&quot;:{&quot;following&quot;:false},&quot;features&quot;:[]}},&quot;reason&quot;:{&quot;_type&quot;:&quot;comment_reason&quot;,&quot;context_url&quot;:&quot;https://genius.com/8846441/Genius-how-genius-works/More-on-annotations&quot;,&quot;display_character&quot;:&quot;R&quot;,&quot;handle&quot;:&quot;Restates the line&quot;,&quot;id&quot;:1,&quot;name&quot;:&quot;restates-the-line&quot;,&quot;raw_name&quot;:&quot;restates the line&quot;,&quot;requires_body&quot;:false,&quot;slug&quot;:&quot;restates_the_line&quot;}},&quot;top_comment&quot;:null,&quot;verified_by&quot;:null}]},&quot;featured_artists&quot;:[{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/3516&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/1ojo0f97s1nnaf0huvp723s75.280x365x1.jpg&quot;,&quot;id&quot;:3516,&quot;image_url&quot;:&quot;https://images.genius.com/1ojo0f97s1nnaf0huvp723s75.280x365x1.jpg&quot;,&quot;index_character&quot;:&quot;f&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Fatman Scoop&quot;,&quot;slug&quot;:&quot;Fatman-scoop&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Fatman-scoop&quot;},{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1630&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/4d9c5ad18c6f8799fa8ea665c7f3ff92.1000x1000x1.png&quot;,&quot;id&quot;:1630,&quot;image_url&quot;:&quot;https://images.genius.com/661ba9186ca36d1226fa7a1cc9215331.1000x1000x1.png&quot;,&quot;index_character&quot;:&quot;c&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Ciara&quot;,&quot;slug&quot;:&quot;Ciara&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Ciara&quot;}],&quot;lyrics_marked_complete_by&quot;:null,&quot;media&quot;:[{&quot;provider&quot;:&quot;youtube&quot;,&quot;start&quot;:0,&quot;type&quot;:&quot;video&quot;,&quot;url&quot;:&quot;http://www.youtube.com/watch?v=HrE5lrjBryI&quot;},{&quot;native_uri&quot;:&quot;spotify:track:0UaMYEvWZi0ZqiDOoHU3YI&quot;,&quot;provider&quot;:&quot;spotify&quot;,&quot;type&quot;:&quot;audio&quot;,&quot;url&quot;:&quot;https://open.spotify.com/track/0UaMYEvWZi0ZqiDOoHU3YI&quot;}],&quot;primary_artist&quot;:{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1529&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/fc183acc7f3e70189a76afe4a0d8cdf7.240x320x1.jpg&quot;,&quot;id&quot;:1529,&quot;image_url&quot;:&quot;https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg&quot;,&quot;index_character&quot;:&quot;m&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Missy Elliott&quot;,&quot;slug&quot;:&quot;Missy-elliott&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Missy-elliott&quot;},&quot;primary_tag&quot;:{&quot;_type&quot;:&quot;tag&quot;,&quot;id&quot;:1434,&quot;name&quot;:&quot;Rap&quot;,&quot;primary&quot;:true,&quot;url&quot;:&quot;https://genius.com/tags/rap&quot;},&quot;producer_artists&quot;:[{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1529&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/fc183acc7f3e70189a76afe4a0d8cdf7.240x320x1.jpg&quot;,&quot;id&quot;:1529,&quot;image_url&quot;:&quot;https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg&quot;,&quot;index_character&quot;:&quot;m&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Missy Elliott&quot;,&quot;slug&quot;:&quot;Missy-elliott&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Missy-elliott&quot;}],&quot;song_relationships&quot;:[{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;samples&quot;,&quot;songs&quot;:[{&quot;_type&quot;:&quot;song&quot;,&quot;annotation_count&quot;:1,&quot;api_path&quot;:&quot;/songs/1977813&quot;,&quot;full_title&quot;:&quot;Clear by Cybotron&quot;,&quot;header_image_thumbnail_url&quot;:&quot;https://images.genius.com/c8d3bfce8903e154125f03ca68a7adb1.300x300x1.jpg&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/c8d3bfce8903e154125f03ca68a7adb1.600x600x1.jpg&quot;,&quot;id&quot;:1977813,&quot;instrumental&quot;:false,&quot;lyrics_owner_id&quot;:1549345,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;lyrics_updated_at&quot;:1432932148,&quot;path&quot;:&quot;/Cybotron-clear-lyrics&quot;,&quot;pyongs_count&quot;:1,&quot;song_art_image_thumbnail_url&quot;:&quot;https://images.genius.com/c8d3bfce8903e154125f03ca68a7adb1.300x300x1.jpg&quot;,&quot;stats&quot;:{&quot;hot&quot;:false,&quot;unreviewed_annotations&quot;:1},&quot;title&quot;:&quot;Clear&quot;,&quot;title_with_featured&quot;:&quot;Clear&quot;,&quot;updated_by_human_at&quot;:1493837879,&quot;url&quot;:&quot;https://genius.com/Cybotron-clear-lyrics&quot;,&quot;primary_artist&quot;:{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/130239&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/86ec198038a746592ca28ca7f58f8f41.480x360x1.jpg&quot;,&quot;id&quot;:130239,&quot;image_url&quot;:&quot;https://images.genius.com/64a7170769057006652b21d5f0120d13.360x360x1.jpg&quot;,&quot;index_character&quot;:&quot;c&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Cybotron&quot;,&quot;slug&quot;:&quot;Cybotron&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Cybotron&quot;}}]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;sampled_in&quot;,&quot;songs&quot;:[{&quot;_type&quot;:&quot;song&quot;,&quot;annotation_count&quot;:1,&quot;api_path&quot;:&quot;/songs/1993120&quot;,&quot;full_title&quot;:&quot;Give and go by Girl Talk&quot;,&quot;header_image_thumbnail_url&quot;:&quot;https://images.genius.com/2ee0c511ce3c4e209b56b579a5a25faf.300x300x1.jpg&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/2ee0c511ce3c4e209b56b579a5a25faf.1000x1000x1.jpg&quot;,&quot;id&quot;:1993120,&quot;instrumental&quot;:false,&quot;lyrics_owner_id&quot;:1549345,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;lyrics_updated_at&quot;:1432938921,&quot;path&quot;:&quot;/Girl-talk-give-and-go-lyrics&quot;,&quot;pyongs_count&quot;:null,&quot;song_art_image_thumbnail_url&quot;:&quot;https://images.genius.com/2ee0c511ce3c4e209b56b579a5a25faf.300x300x1.jpg&quot;,&quot;stats&quot;:{&quot;hot&quot;:false,&quot;unreviewed_annotations&quot;:0},&quot;title&quot;:&quot;Give and go&quot;,&quot;title_with_featured&quot;:&quot;Give and go&quot;,&quot;updated_by_human_at&quot;:1532443935,&quot;url&quot;:&quot;https://genius.com/Girl-talk-give-and-go-lyrics&quot;,&quot;primary_artist&quot;:{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/928&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/d8d897c8aa4269e0035eb18faffd52ce.770x616x1.jpg&quot;,&quot;id&quot;:928,&quot;image_url&quot;:&quot;https://images.genius.com/5951c5c6da66bc7fc4ae35c95a578766.1000x1000x1.jpg&quot;,&quot;index_character&quot;:&quot;g&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Girl Talk&quot;,&quot;slug&quot;:&quot;Girl-talk&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Girl-talk&quot;}},{&quot;_type&quot;:&quot;song&quot;,&quot;annotation_count&quot;:0,&quot;api_path&quot;:&quot;/songs/2379726&quot;,&quot;full_title&quot;:&quot;I Fucking Bleed Purple And Gold by Super Mash Bros.&quot;,&quot;header_image_thumbnail_url&quot;:&quot;https://images.rapgenius.com/d078d74dbc5e5c3f6c2b62d33427568a.300x300x1.jpg&quot;,&quot;header_image_url&quot;:&quot;https://images.rapgenius.com/d078d74dbc5e5c3f6c2b62d33427568a.600x600x1.jpg&quot;,&quot;id&quot;:2379726,&quot;instrumental&quot;:false,&quot;lyrics_owner_id&quot;:2381354,&quot;lyrics_state&quot;:&quot;complete&quot;,&quot;lyrics_updated_at&quot;:1449382193,&quot;path&quot;:&quot;/Super-mash-bros-i-fucking-bleed-purple-and-gold-lyrics&quot;,&quot;pyongs_count&quot;:null,&quot;song_art_image_thumbnail_url&quot;:&quot;https://images.rapgenius.com/d078d74dbc5e5c3f6c2b62d33427568a.300x300x1.jpg&quot;,&quot;stats&quot;:{&quot;hot&quot;:false,&quot;unreviewed_annotations&quot;:0},&quot;title&quot;:&quot;I Fucking Bleed Purple And Gold&quot;,&quot;title_with_featured&quot;:&quot;I Fucking Bleed Purple And Gold&quot;,&quot;updated_by_human_at&quot;:1528197337,&quot;url&quot;:&quot;https://genius.com/Super-mash-bros-i-fucking-bleed-purple-and-gold-lyrics&quot;,&quot;primary_artist&quot;:{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/613857&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/e56dafb0e969e7467cb7caf0ee564aff.100x100x1.jpg&quot;,&quot;id&quot;:613857,&quot;image_url&quot;:&quot;https://images.genius.com/e56dafb0e969e7467cb7caf0ee564aff.100x100x1.jpg&quot;,&quot;index_character&quot;:&quot;s&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Super Mash Bros.&quot;,&quot;slug&quot;:&quot;Super-mash-bros&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Super-mash-bros&quot;}}]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;interpolates&quot;,&quot;songs&quot;:[]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;interpolated_by&quot;,&quot;songs&quot;:[]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;cover_of&quot;,&quot;songs&quot;:[]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;covered_by&quot;,&quot;songs&quot;:[]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;remix_of&quot;,&quot;songs&quot;:[]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;remixed_by&quot;,&quot;songs&quot;:[]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;live_version_of&quot;,&quot;songs&quot;:[]},{&quot;_type&quot;:&quot;song_relationship&quot;,&quot;type&quot;:&quot;performed_live_as&quot;,&quot;songs&quot;:[]}],&quot;tags&quot;:[{&quot;_type&quot;:&quot;tag&quot;,&quot;id&quot;:1434,&quot;name&quot;:&quot;Rap&quot;,&quot;primary&quot;:true,&quot;url&quot;:&quot;https://genius.com/tags/rap&quot;}],&quot;top_scholar&quot;:{&quot;_type&quot;:&quot;user_attribution&quot;,&quot;attribution_value&quot;:66.0,&quot;pinned_role&quot;:null,&quot;user&quot;:{&quot;_type&quot;:&quot;user&quot;,&quot;about_me_summary&quot;:&quot;https://genius.com/discussions/78086-Ben-devlin-january-april-2014-bandcamp-compilation\n\nhttps://www.youtube.com/watch?v=Kmys4LH9jTE\n\n\n\n“1234; hardworking and fair, bizarre”\n\n\n\n\n\nI’m somewhere quite high on here\n\nRIP Mary Devlin\n04/06/1943 – 14/03/2013\n\nIn no particular order, my favourite 10 producers are;\n\n\n\nDJ Premier\n\n\nEl-P\n\n\nJ Dilla\n\n\nKanye West\n\n\nMadlib\n\n\nPete Rock\n\n\nRZA\n\n\nThe Neptunes\n\n\nTimbaland\n\n\nWiley\n\n\n\n\n#GOAT song ever\n\nChris: Today’s historic trade agreement between Australia and Hong Kong marks a new season of hope for the future of world trade. The two countries have been at each other’s throats for years but now the hatchet’s been buried by a treaty which allows unrestricted trading between all parties at all levels. I’m joined now by Martin Craste, the British minister with special responsibility for the Commonwealth, and Gavin Hawtrey, the Australian foreign secretary, on camera. Gentlemen, this is pretty historic stuff, well done. A future of unbridled harmony then, Australia?\n\nGavin: Yes, I think Martin Craste and I can be pretty satisfied. It’s a good day\n\nChris: And if, as in the past, Australia exceed their agreement, what will you do about it?\n\nMartin: This is a very satisfactory treaty which I’m sure will work, well naturally if the limits were exceeded then this would be met with a firm line but I can’t see this being necessary\n\nChris: Mr. Hawtrey, he’s knocking a firm line in your direction. What are you going to do about that?\n\nGavin: Well, in that case we’d just reimpose sanctions as we did last year and then we’d-\n\nChris: Sanctions? Hold on a second, they’ve only just swallowed their sanctions and now they’re burping them back up in your face\n\nMartin: I think sanctions is rather premature talk, certainly. If sanctions were imposed we should have to retaliate with appropriate measures, but I can’t see-\n\nChris: I think “appropriate measures” is a euphemism, Mr. Hawtrey. You know what it means, what are you going to do about that?\n\nGavin: Well, I’d just have to go back to cabinet\n\nChris: And ask them about what?\n\nGavin: Well, I don’t know. Maybe it’s a matter for the military\n\nChris: The military?\n\nMartin: I think military measures is totally inappropriate reaction and I think this is way, way over the top\n\nChris: Sounds like you’re being inappropriate, are you?\n\nGavin: Of course I’m not being inappropriate, Martin Craste knows that full well\n\nMartin: This is the sort of misunderstanding that I thought we’d laid to rest during our negotiating period\n\nChris: Misunderstanding it certainly is, it’s certainly not a treaty, is it? You’re both at each other’s throats, you’re backing yourselfs up with arms, what are you going to do about it? Mr. Hawtrey, let me give you a hint; bang\n\nGavin: What are you asking me to say?\n\nChris: You know damn well what I’m asking you to say. You’re putting yourself in a situation of armed conflict, what are you plunging yourself into?\n\nGavin: You’d like me to say it?\n\nChris: I want you to say it, yes\n\nGavin: You want the word?\n\nChris: The word\n\nGavin: I will not flinch-\n\nChris: You will not flinch from?\n\nGavin: War\n\nChris: War. Gentlemen, I’ll put you on hold. If fighting did break out it would probably occur in Eastman’s Town in the upper cataracts on the Australio-Hong Kong border. Our reporter Donald Bethlehem is there now; Donald, what’s the atmosphere like?\n\nDonald: Tension here is very high, Chris. The stretched twig of peace is at melting point, people here are literally bursting with war. This is very much a country that’s going to blow up in its face\n\nChris: Well, gentlemen, it seems we have little option now but to declare war immediately\n\nMartin: Well, this is quite impossible. I couldn’t take such a decision without referring to my superior, Chris Patten, he’s in Hong Kong\n\nChris: Good, because he’s on the line now via satellite. Mr. Patten, what do you think of the idea of a war now? … I’ll take that as a yes\n\nMartin: Very well, it’s war\n\nGavin: War it is\n\nDonald: That’s it, Chris. It’s war, war has broken out, this is a war\n\nChris: That’s it, yes, it’s war\n\n#1 IANAHB2 scholar\n\nFavourite 5 albums:\n- Fever\n- Boy In Da Corner\n- Ultravisitor\n- YoYoYoYoYo\n- Paul’s Boutique\n\nShout out to HailTheKing, who editored me, I think it was March 2nd 2013 but can’t be sure&quot;,&quot;api_path&quot;:&quot;/users/141291&quot;,&quot;avatar&quot;:{&quot;tiny&quot;:{&quot;url&quot;:&quot;https://images.rapgenius.com/avatars/tiny/845fd01203157969bce194cbc8065f03&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:16,&quot;height&quot;:16}},&quot;thumb&quot;:{&quot;url&quot;:&quot;https://images.rapgenius.com/avatars/thumb/845fd01203157969bce194cbc8065f03&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:32,&quot;height&quot;:32}},&quot;small&quot;:{&quot;url&quot;:&quot;https://images.rapgenius.com/avatars/small/845fd01203157969bce194cbc8065f03&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:100,&quot;height&quot;:100}},&quot;medium&quot;:{&quot;url&quot;:&quot;https://images.rapgenius.com/avatars/medium/845fd01203157969bce194cbc8065f03&quot;,&quot;bounding_box&quot;:{&quot;width&quot;:300,&quot;height&quot;:400}}},&quot;header_image_url&quot;:&quot;https://images.rapgenius.com/avatars/medium/845fd01203157969bce194cbc8065f03&quot;,&quot;human_readable_role_for_display&quot;:&quot;Contributor&quot;,&quot;id&quot;:141291,&quot;iq&quot;:83233,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;login&quot;:&quot;123andtotha4&quot;,&quot;name&quot;:&quot;123andtotha4&quot;,&quot;role_for_display&quot;:&quot;contributor&quot;,&quot;url&quot;:&quot;https://genius.com/123andtotha4&quot;,&quot;current_user_metadata&quot;:{&quot;permissions&quot;:[],&quot;excluded_permissions&quot;:[&quot;follow&quot;],&quot;interactions&quot;:{&quot;following&quot;:false},&quot;features&quot;:[]}}},&quot;verified_annotations_by&quot;:[],&quot;verified_contributors&quot;:[],&quot;verified_lyrics_by&quot;:[],&quot;writer_artists&quot;:[{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/994247&quot;,&quot;header_image_url&quot;:&quot;https://assets.genius.com/images/default_avatar_300.png?1544632201&quot;,&quot;id&quot;:994247,&quot;image_url&quot;:&quot;https://assets.genius.com/images/default_avatar_300.png?1544632201&quot;,&quot;index_character&quot;:&quot;c&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Curtis Hudson&quot;,&quot;slug&quot;:&quot;Curtis-hudson&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Curtis-hudson&quot;},{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/130240&quot;,&quot;header_image_url&quot;:&quot;https://assets.genius.com/images/default_avatar_300.png?1544632201&quot;,&quot;id&quot;:130240,&quot;image_url&quot;:&quot;https://assets.genius.com/images/default_avatar_300.png?1544632201&quot;,&quot;index_character&quot;:&quot;r&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Richard Davis&quot;,&quot;slug&quot;:&quot;Richard-davis&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Richard-davis&quot;},{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/3516&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/1ojo0f97s1nnaf0huvp723s75.280x365x1.jpg&quot;,&quot;id&quot;:3516,&quot;image_url&quot;:&quot;https://images.genius.com/1ojo0f97s1nnaf0huvp723s75.280x365x1.jpg&quot;,&quot;index_character&quot;:&quot;f&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Fatman Scoop&quot;,&quot;slug&quot;:&quot;Fatman-scoop&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Fatman-scoop&quot;},{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/3486&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/972b9476d154761179133dca5069ec93.413x480x1.jpg&quot;,&quot;id&quot;:3486,&quot;image_url&quot;:&quot;https://images.genius.com/972b9476d154761179133dca5069ec93.413x480x1.jpg&quot;,&quot;index_character&quot;:&quot;g&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Grand Puba&quot;,&quot;slug&quot;:&quot;Grand-puba&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Grand-puba&quot;},{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/183&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/b32cb1cd5c81877758601ed10eb421e2.549x549x1.jpg&quot;,&quot;id&quot;:183,&quot;image_url&quot;:&quot;https://images.genius.com/b32cb1cd5c81877758601ed10eb421e2.549x549x1.jpg&quot;,&quot;index_character&quot;:&quot;j&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Ja Rule&quot;,&quot;slug&quot;:&quot;Ja-rule&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Ja-rule&quot;},{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1630&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/4d9c5ad18c6f8799fa8ea665c7f3ff92.1000x1000x1.png&quot;,&quot;id&quot;:1630,&quot;image_url&quot;:&quot;https://images.genius.com/661ba9186ca36d1226fa7a1cc9215331.1000x1000x1.png&quot;,&quot;index_character&quot;:&quot;c&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Ciara&quot;,&quot;slug&quot;:&quot;Ciara&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Ciara&quot;},{&quot;_type&quot;:&quot;artist&quot;,&quot;api_path&quot;:&quot;/artists/1529&quot;,&quot;header_image_url&quot;:&quot;https://images.genius.com/fc183acc7f3e70189a76afe4a0d8cdf7.240x320x1.jpg&quot;,&quot;id&quot;:1529,&quot;image_url&quot;:&quot;https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg&quot;,&quot;index_character&quot;:&quot;m&quot;,&quot;is_meme_verified&quot;:false,&quot;is_verified&quot;:false,&quot;name&quot;:&quot;Missy Elliott&quot;,&quot;slug&quot;:&quot;Missy-elliott&quot;,&quot;url&quot;:&quot;https://genius.com/artists/Missy-elliott&quot;}]}}" itemprop="page_data"/>
    </meta></link></meta></meta></meta></head>
    <body class="act-show cont-songs snarly">
    <preload ng-non-bindable=""><preload-content data-preload_data='{"iq_by_event_type":{"accepted_a_lyrics_edit":3.0,"annotation_downvote_by_contrib":-1.0,"annotation_downvote_by_default":-1.0,"annotation_downvote_by_editor":-1.0,"annotation_downvote_by_high_iq_user":-1.0,"annotation_downvote_by_moderator":-1.0,"annotation_upvote_by_contrib":4.0,"annotation_upvote_by_default":2.0,"annotation_upvote_by_editor":6.0,"annotation_upvote_by_high_iq_user":4.0,"annotation_upvote_by_moderator":10.0,"answer_downvote_by_contrib":-1.0,"answer_downvote_by_default":-1.0,"answer_downvote_by_editor":-1.0,"answer_downvote_by_high_iq_user":-1.0,"answer_downvote_by_moderator":-1.0,"answered_a_question":5.0,"answered_a_question_meme":25.0,"answered_a_question_verified":10.0,"answer_upvote_by_contrib":4.0,"answer_upvote_by_default":2.0,"answer_upvote_by_editor":6.0,"answer_upvote_by_high_iq_user":4.0,"answer_upvote_by_moderator":10.0,"archived_a_question":1.0,"article_downvote":-1.0,"article_upvote":1.0,"asked_a_question":1.0,"auto_accepted_explanation":15.0,"comment_downvote":-0.5,"comment_upvote":0.5,"created_a_lyrics_edit":2.0,"created_a_real_song":40.0,"created_a_song":5.0,"forum_post_downvote":-0.5,"forum_post_upvote":0.5,"historical_you_published_a_song":60.0,"metadata_update_or_addition":2.0,"pending_explanation":5.0,"pinned_a_question_not_your_own":2.0,"question_downvote":-2.0,"question_upvote":2.0,"rejected_a_lyrics_edit":2.0,"song_metadata_update_or_addition":2.0,"song_pageviews_1000":25.0,"song_pageviews_10000":50.0,"song_pageviews_100000":125.0,"song_pageviews_1000000":500.0,"song_pageviews_2500":30.0,"song_pageviews_25000":75.0,"song_pageviews_250000":150.0,"song_pageviews_2500000":1000.0,"song_pageviews_500":20.0,"song_pageviews_5000":35.0,"song_pageviews_50000":100.0,"song_pageviews_500000":200.0,"song_pageviews_5000000":2000.0,"suggestion_downvote_by_contrib":-1.0,"suggestion_downvote_by_default":-0.5,"suggestion_downvote_by_editor":-1.0,"suggestion_downvote_by_high_iq_user":-1.0,"suggestion_downvote_by_moderator":-1.0,"suggestion_upvote_by_contrib":2.0,"suggestion_upvote_by_default":1.0,"suggestion_upvote_by_editor":3.0,"suggestion_upvote_by_high_iq_user":2.0,"suggestion_upvote_by_moderator":4.0,"verified_explanation_by_meme":100.0,"verified_explanation_by_non_meme":15.0,"verified_lyrics_by_meme_featured":50.0,"verified_lyrics_by_meme_primary":75.0,"verified_lyrics_by_meme_writer":75.0,"verified_lyrics_by_non_meme_featured":10.0,"verified_lyrics_by_non_meme_primary":15.0,"verified_lyrics_by_non_meme_writer":15.0,"you_accepted_a_comment":6.0,"you_accepted_an_annotation":10.0,"you_added_a_photo":100.0,"you_archived_a_comment":2.0,"you_deleted_an_annotation":5.0,"you_incorporated_an_annotation":5.0,"you_integrated_a_comment":2.0,"you_linked_an_identity":100.0,"you_merged_an_annotation_edit":4.0,"you_published_a_song":5.0,"your_annotation_accepted":10.0,"your_annotation_edit_merged":5.0,"your_annotation_edit_rejected":-0.5,"your_annotation_incorporated":15.0,"your_annotation_rejected":0.0,"your_annotation_was_cosigned_by_community_verified":2.0,"your_annotation_was_cosigned_by_meme":50.0,"your_annotation_was_cosigned_by_verified_verified":20.0,"your_answer_cleared":-5.0,"your_answer_pinned":5.0,"your_comment_accepted":2.0,"your_comment_archived":0.0,"your_comment_integrated":2.0,"your_comment_rejected":-0.5,"you_rejected_a_comment":2.0,"you_rejected_an_annotation":2.0,"you_rejected_an_annotation_edit":2.0,"your_lyrics_edit_accepted":3.0,"your_lyrics_edit_rejected":-2.0,"your_question_answered":4.0,"your_question_archived":-1.0,"your_question_pinned":5.0}}'></preload-content></preload>
    <div class="header" click-outside="close_mobile_subnav_menu()" ng-controller="HeaderCtrl as header_ctrl">
    <global-message ng-cloak="" ng-if="header_ctrl.cloudflare_error">
        {{:: 'cloud_flare_always_on_short_message' | i18n }}
        <br/>Check <a href="https://twitter.com/genius" target="_blank">@genius</a> for updates. We'll have things fixed soon.
      </global-message>
    <div class="header-primary active">
    <div class="header-expand_nav_menu" ng-click="toggle_mobile_subnav_menu()"><div class="header-expand_nav_menu-contents"></div></div>
    <div class="logo_container">
    <a class="logo_link" href="https://genius.com/">GENIUS</a>
    </div>
    <header-actions></header-actions>
    </div>
    <search-form search-style="header"></search-form>
    <ul class="header-nav_menu" ng-class="{'header-nav_menu--visible': mobile_subnav_menu_open}">
    <div ng-cloak="">
    <header-menu-item class="header-nav_menu--hide_when_expand_menu_is_not_available" name="Home" url="https://genius.com/"></header-menu-item>
    <header-menu-item name="Featured Stories" url="https://genius.com/#featured-stories"></header-menu-item>
    <header-menu-item name="Top Songs" url="https://genius.com/#top-songs"></header-menu-item>
    <header-menu-item name="Videos" url="https://genius.com/#videos"></header-menu-item>
    <header-menu-item name="Community" url="https://genius.com/#community"></header-menu-item>
    <span class="nav_menu-item nav_menu-item--separator">|</span>
    <header-menu-item name="Shop" url="https://shop.genius.com/"></header-menu-item>
    <span class="nav_menu-item nav_menu-item--separator">|</span>
    <li class="nav_menu-item">
    <a class="nav_menu-link nav_menu-link--facebook" href="https://www.facebook.com/geniusdotcom/" target="_blank">
    <svg class="inline_icon" src="facebook.svg"></svg>
    <span class="nav_menu-link-social_text">Facebook</span>
    </a>
    </li>
    <li class="nav_menu-item">
    <a class="nav_menu-link" href="https://twitter.com/Genius" target="_blank">
    <svg class="inline_icon" src="twitter.svg"></svg>
    <span class="nav_menu-link-social_text">Twitter</span>
    </a>
    </li>
    <li class="nav_menu-item">
    <a class="nav_menu-link" href="https://www.instagram.com/genius/" target="_blank">
    <svg class="inline_icon" src="instagram.svg"></svg>
    <span class="nav_menu-link-social_text">Instagram</span>
    </a>
    </li>
    <li class="nav_menu-item">
    <a class="nav_menu-link nav_menu-link--no_right_padding" href="https://www.youtube.com/genius" target="_blank">
    <svg class="inline_icon nav_menu-link-youtube_icon" src="youtube.svg"></svg>
    <span class="nav_menu-link-social_text">Youtube</span>
    </a>
    </li>
    </div>
    </ul>
    </div>
    <div class="global_messages">
    <flash-messages></flash-messages>
    </div>
    <script type="application/ld+json">
            {"@context":"http://schema.org","@type":"MusicRecording","byArtist":{"@context":"http://schema.org","@type":"MusicGroup","name":"Missy Elliott","url":"https://genius.com/artists/Missy-elliott","description":"Melissa Arnette Elliott, sometimes known Missy “Misdemeanor” Elliott, or just simply Missy Elliot, was born on July 1, 1971. Many consider Missy Elliott as the best female rapper","image":"https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg"},"image":"https://images.genius.com/51639e476eac5cb5bed5794c023763ff.700x707x1.jpg","inAlbum":[{"@context":"http://schema.org","@type":"MusicAlbum","byArtist":{"@context":"http://schema.org","@type":"MusicGroup","name":"Missy Elliott","url":"https://genius.com/artists/Missy-elliott","description":"Melissa Arnette Elliott, sometimes known Missy “Misdemeanor” Elliott, or just simply Missy Elliot, was born on July 1, 1971. Many consider Missy Elliott as the best female rapper","image":"https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg"},"image":"https://images.genius.com/49c23ba7107cf9bb93749df01b30111c.1000x1000x1.jpg","name":"The Cookbook","url":"https://genius.com/albums/Missy-elliott/The-cookbook","datePublished":"2005-07-04","numTracks":17},{"@context":"http://schema.org","@type":"MusicAlbum","byArtist":{"@context":"http://schema.org","@type":"MusicGroup","name":"Missy Elliott","url":"https://genius.com/artists/Missy-elliott","description":"Melissa Arnette Elliott, sometimes known Missy “Misdemeanor” Elliott, or just simply Missy Elliot, was born on July 1, 1971. Many consider Missy Elliott as the best female rapper","image":"https://images.genius.com/89d1c14239087451a1f363fa21e2525d.768x768x1.jpg"},"image":"https://images.genius.com/75cfb85ca9e5a85370891a511a97b3d5.640x640x1.jpg","name":"Respect M.E.","url":"https://genius.com/albums/Missy-elliott/Respect-m-e","datePublished":"2006-09-04","numTracks":17}],"name":"Lose Control","url":"https://genius.com/Missy-elliott-lose-control-lyrics","datePublished":"2005-05-27"}
          </script>
    <routable-page>
    <ng-non-bindable>
    <div class="leaderboard_ad_container">
    <div class="dfp_unit u-vertical_margins dfp_unit--centered_billboard">
    <div style="
          width: 728px;
          height: 90px;
        "></div>
    </div>
    </div>
    <div class="header_with_cover_art">
    <div class="header_with_cover_art-inner column_layout">
    <div class="column_layout-column_span column_layout-column_span--primary">
    <div class="header_with_cover_art-cover_art ">
    <div class="cover_art">
    <img alt="Https%3a%2f%2fimages" class="cover_art-image" src="https://t2.genius.com/unsafe/220x223/https%3A%2F%2Fimages.genius.com%2F51639e476eac5cb5bed5794c023763ff.700x707x1.jpg" srcset="https://t2.genius.com/unsafe/440x445/https%3A%2F%2Fimages.genius.com%2F51639e476eac5cb5bed5794c023763ff.700x707x1.jpg 2x"/>
    </div>
    </div>
    <div class="header_with_cover_art-primary_info_container">
    <div class="header_with_cover_art-primary_info">
    <h1 class="header_with_cover_art-primary_info-title ">Lose Control</h1>
    <h2>
    <a class="header_with_cover_art-primary_info-primary_artist" href="https://genius.com/artists/Missy-elliott">Missy Elliott</a>
    </h2>
    <h3>
    <div class="metadata_unit ">
    <span class="metadata_unit-label">Featuring</span>
    <span class="metadata_unit-info">
    <a href="https://genius.com/artists/Fatman-scoop">Fatman Scoop</a> &amp; <a href="https://genius.com/artists/Ciara">Ciara</a>
    </span>
    </div>
    </h3>
    <h3>
    <div class="metadata_unit ">
    <span class="metadata_unit-label">Produced by</span>
    <span class="metadata_unit-info">
    <a href="https://genius.com/artists/Missy-elliott">Missy Elliott</a>
    </span>
    </div>
    </h3>
    <h3>
    <div class="metadata_unit ">
    <span class="metadata_unit-label">Album</span>
    <span class="metadata_unit-info"><a href="https://genius.com/albums/Missy-elliott/The-cookbook">The Cookbook</a></span>
    </div>
    </h3>
    </div>
    </div>
    </div>
    <div class="column_layout-column_span column_layout-column_span--secondary u-top_margin">
    </div>
    </div>
    </div>
    <div class="song_body column_layout" initial-content-for="song_body">
    <div class="column_layout-column_span column_layout-column_span--primary">
    <div class="song_body-lyrics">
    <h2 class="text_label text_label--gray text_label--x_small_text_size u-top_margin">Lose Control Lyrics</h2>
    <div initial-content-for="lyrics">
    <div class="lyrics">
    <!--sse-->
    <p>[Intro: Fatman Scoop]<br/>
    <a annotation-fragment="1760210" class="referent" classification="accepted" data-id="1760210" href="/Missy-elliott-lose-control-lyrics#note-1760210" image="false" ng-class="{
              'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),
              'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),
              'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)
            }" ng-click="open()" on-hover-with-no-digest="set_current_hover_and_digest(hover ? fragment_id : undefined)" pending-editorial-actions-count="0" prevent-default-click=""><i>Music make you lose control, music make you lose control</i></a><br/>
    Let's go! Hey, hey, hey, hey, hey, hey<br/>
    Here we go now, here we go now, here we go now, here we go now<br/>
    <a annotation-fragment="8681095" class="referent" classification="accepted" data-id="8681095" href="/Missy-elliott-lose-control-lyrics#note-8681095" image="false" ng-class="{
              'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),
              'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),
              'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)
            }" ng-click="open()" on-hover-with-no-digest="set_current_hover_and_digest(hover ? fragment_id : undefined)" pending-editorial-actions-count="0" prevent-default-click="">(Music make you lose control)</a><br/>
    Misdemeanor's in the house<br/>
    Ciara's in the house<br/>
    Misdemeanor's in the house<br/>
    Fatman Scoop-man Scoop-man Scoop..<br/>
    <br/>
    [Verse 1: Missy]<br/>
    <a annotation-fragment="4307282" class="referent" classification="accepted" data-id="4307282" href="/Missy-elliott-lose-control-lyrics#note-4307282" image="false" ng-class="{
              'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),
              'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),
              'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)
            }" ng-click="open()" on-hover-with-no-digest="set_current_hover_and_digest(hover ? fragment_id : undefined)" pending-editorial-actions-count="0" prevent-default-click="">I got a cute face, chubby waist</a><br/>
    Thick legs, in shape<br/>
    Rump shakin, both ways<br/>
    Make you do a double take<br/>
    <a annotation-fragment="1760219" class="referent" classification="accepted" data-id="1760219" href="/Missy-elliott-lose-control-lyrics#note-1760219" image="false" ng-class="{
              'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),
              'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),
              'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)
            }" ng-click="open()" on-hover-with-no-digest="set_current_hover_and_digest(hover ? fragment_id : undefined)" pending-editorial-actions-count="0" prevent-default-click="">Planet rocker,</a> show stopper<br/>
    Flow proper, head knocker<br/>
    Beat scholar, tail dropper<br/>
    Do my thang, motherfucker<br/>
    My Rolls Royce, Lamborghini<br/>
    Blue Madena, always beamin<br/>
    Rag top, chrome pipes<br/>
    Blue lights, outta sight<br/>
    (Long weave) sewed in<br/>
    (Say it again) sewed in<br/>
    Make that money, throw it in<br/>
    Booty bouncin, gone head<br/>
    <br/>
    [Hook: Ciara &amp; Missy Elliot]<br/>
    Everybody here - get it out of control<br/>
    Get your backs off the wall, cause <a annotation-fragment="3480729" class="referent" classification="unreviewed" data-id="3480729" href="/Missy-elliott-lose-control-lyrics#note-3480729" image="false" ng-class="{
              'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),
              'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),
              'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)
            }" ng-click="open()" on-hover-with-no-digest="set_current_hover_and_digest(hover ? fragment_id : undefined)" pending-editorial-actions-count="0" prevent-default-click="">Misdemeanor</a> said so<br/>
    Everybody, everybody, everybody, everybody<br/>
    (Just throw your hands in the air!)<br/>
    <br/>
    [Verse 2: Ciara &amp; (Missy Elliot)]<br/>
    Well my name is Ciara, for all you fly fellas<br/>
    No one, can do it better (she'll sing on acapella)<br/>
    Boy the music makes me lose control<br/>
    <br/>
    [Verse 3: Missy]<br/>
    (Now bring it back now!) We gon' make you lose control<br/>
    And let it go, 'fore you know, you gon' hit the flo'<br/>
    I rock to the beat til I'm (tired)<br/>
    I walk in the club it's (fire)<br/>
    Get it crunk and wired<br/>
    Wave your hands scream (louder)<br/>
    If you smoke then fire it up<br/>
    Brang the roof down and (holla)<br/>
    If you tipsy stand up<br/>
    DJ turn it (louder)<br/>
    Take somebody by the waist and (uhh!)<br/>
    Now throw it in they face like (uhh!)<br/>
    Hypnotic robotic, this here will rock yo' bodies<br/>
    Take somebody by the waist and (uhh!)<br/>
    Now throw it in they face like (uhh!)<br/>
    Systematic ecstatic (THIS HIT BE AUTOMATIC)<br/>
    <br/>
    [Bridge - Missy &amp; Fatman Scoop]<br/>
    Work me, work, work<br/>
    Work me, work, work<br/>
    Work me, work, work<br/>
    Work me, do it right<br/>
    Hit the floor, hit the floor<br/>
    Hit the floor, hit the floor<br/>
    Hit the floor, hit the floor<br/>
    Hit the floor, hit the floor<br/>
    <br/>
    [Hook - Missy &amp; Fatman]<br/>
    Everybody here - get it out of control<br/>
    Get your backs off the wall, cause Misdemeanor said so<br/>
    Everybody, everybody, everybody, everybody<br/>
    (Just throw your hands in the air!)<br/>
    Everybody here - get it out of control<br/>
    Get your backs off the wall, cause Misdemeanor said so<br/>
    Everybody, everybody, everybody, everybody<br/>
    (Just throw your hands in the air!)<br/>
    <br/>
    [Fatman Scoop] + (Missy)<br/>
    Get your back off the wall, get your back off the wall<br/>
    Get your back off the wall, get your back off the wall<br/>
    (Everybody, get loose) Now put your back, on the wall<br/>
    Put your back, on the wall<br/>
    Put your back, on the wall, put your back, on the wall<br/>
    Misdemeanor's in the house<br/>
    Yeah, Ciara's in the house<br/>
    Misdemeanor's in the house, "Music make you lose control"<br/>
    We on fire, we on fire, we on fire, we on fire<br/>
    Now throw it girl, throw it girl, throw it girl, yes<br/>
    Now move your arms to the left girl<br/>
    Now move your arms to the left girl<br/>
    Now move your arms to the right girl<br/>
    Now move your arms to the right girl<br/>
    Let's go now, let's go now, let's go now, WOO! Let's go<br/>
    Should I bring it back right now?<br/>
    Now bring it back down!<br/>
    WOO! Oh, I see you <a annotation-fragment="15871347" class="referent" classification="unreviewed" data-id="15871347" href="/Missy-elliott-lose-control-lyrics#note-15871347" image="false" ng-class="{
              'referent--linked_to_preview': song_ctrl.referent_has_preview(fragment_id),
              'referent--linked_to_preview_active': song_ctrl.highlight_preview_referent(fragment_element_id),
              'referent--purple_indicator': song_ctrl.show_preview_referent_indicator(fragment_element_id)
            }" ng-click="open()" on-hover-with-no-digest="set_current_hover_and_digest(hover ? fragment_id : undefined)" pending-editorial-actions-count="0" prevent-default-click="">C</a><br/>
    Now see, I'mma I'mma do it like C do it<br/>
    Now shake it girl, c'mon and just shake it girl<br/>
    C'mon and let it pop right girl, c'mon and let it pop right girl<br/>
    Now, now, now back it up girl, back it up girl<br/>
    Back it up girl, back it up girl<br/>
    WOO! WOO! WOO! Yo, yo<br/>
    Bring it to the front girl, yo, yo<br/>
    Bring it to the front girl, yo, yo<br/>
    Bring it to the front girl, yo, yo<br/>
    Bring it to the front girl, let's go, let's go</p>
    <!--/sse-->
    </div>
    </div>
    <div initial-content-for="recirculated_content">
    <div class="u-xx_large_vertical_margins">
    <div class="text_label text_label--gray">More on Genius</div>
    <a class="recirculated_content" href="https://genius.com/a/ciara-and-missy-elliott-team-up-for-first-time-in-nearly-a-decade-on-level-up-remix">
    <div class="recirculated_content-image"></div>
    <div class="recirculated_content-info">
    <div class="recirculated_content-title">Ciara And Missy Elliott Team Up For First Time In Nearly A Decade On “Level Up” Remix</div>
    </div>
    </a>
    </div>
    </div>
    </div>
    </div>
    <div class="column_layout-column_span column_layout-column_span--secondary u-top_margin column_layout-flex_column">
    <div class="column_layout-column_span-initial_content">
    <div class="dfp_unit u-x_large_bottom_margin dfp_unit--in_read">
    <div style="
          width: 300px;
          height: 250px;
        "></div>
    </div>
    <div class="annotation_label">
    <h3 class="u-inline">
                About “Lose Control”
              </h3>
    </div>
    <div class="rich_text_formatting">
    <p>Another classic Missy Elliott banger, centred around the loss of control of your extremities on the dance floor. It samples Hot Streak’s <a href="https://www.youtube.com/watch?v=HQiL-iHsVu0" rel="noopener nofollow">“Body Work”</a> and Cybotron’s <a data-api_path="/songs/1977813" href="https://genius.com/Cybotron-clear-lyrics" rel="noopener">“Clear,”</a> and produced by Missy herself, featuring Ciara, off the strength of her incredible <a data-api_path="/albums/1561" href="https://genius.com/albums/Ciara/Goodies" rel="noopener"><em>Goodies</em></a> album, and Fatman Scoop as hype man.</p>
    <p>The <a href="https://www.youtube.com/watch?v=khgIVMUvihg" rel="noopener nofollow">video</a> features a bunch of dancers engaging in futuristic moves in various settings, with Missy displaying <a href="https://youtu.be/khgIVMUvihg?t=1m2s" rel="noopener nofollow">cutting edge technology.</a></p>
    <p>The song was a chart smash, making it to number 3 on the <em>US Billboard</em> charts, and top 10 in 4 other countries.</p>
    </div>
    <div class="song_metadata u-xx_large_bottom_margin"></div>
    </div>
    <div class="column_layout-column_span-initial_content">
    <div initial-content-for="question_list">
    <ul>
    </ul>
    </div>
    <div initial-content-for="track_info">
    <div class="u-xx_large_vertical_margins show_tiny_edit_button_on_hover">
    <h3 class="text_label u-x_small_bottom_margin">"Lose Control" Track Info</h3>
    <div class="metadata_unit metadata_unit--table_row">
    <span class="metadata_unit-label">Written By</span>
    <span class="metadata_unit-info">
    <a href="https://genius.com/artists/Curtis-hudson">Curtis Hudson</a>, <a href="https://genius.com/artists/Richard-davis">Richard Davis</a>, <a href="https://genius.com/artists/Fatman-scoop">Fatman Scoop</a> &amp; <span class="metadata_unit-show_more">4 more</span>
    </span>
    </div>
    <div class="metadata_unit metadata_unit--table_row">
    <span class="metadata_unit-label">Label</span>
    <span class="metadata_unit-info">
    <a href="https://genius.com/artists/Atlantic-records">Atlantic Records</a>
    </span>
    </div>
    <div class="metadata_unit metadata_unit--table_row">
    <span class="metadata_unit-label">Release Date</span>
    <span class="metadata_unit-info metadata_unit-info--text_only">May 27, 2005</span>
    </div>
    <div class="metadata_unit metadata_unit--table_row">
    <span class="metadata_unit-label">Samples</span>
    <span class="metadata_unit-info">
    <div class="u-x_small_bottom_margin">
    <a href="https://genius.com/Cybotron-clear-lyrics">Clear by Cybotron</a>
    </div>
    </span>
    </div>
    <div class="metadata_unit metadata_unit--table_row">
    <span class="metadata_unit-label">Sampled In</span>
    <span class="metadata_unit-info">
    <div class="u-x_small_bottom_margin">
    <a href="https://genius.com/Girl-talk-give-and-go-lyrics">Give and go by Girl Talk</a>
    </div>
    <div class="u-x_small_bottom_margin">
    <a href="https://genius.com/Super-mash-bros-i-fucking-bleed-purple-and-gold-lyrics">I Fucking Bleed Purple And Gold by Super Mash Bros.</a>
    </div>
    </span>
    </div>
    </div>
    </div>
    <div initial-content-for="album">
    <div class="u-xx_large_vertical_margins">
    <div class="song_album u-bottom_margin">
    <a class="song_album-album_art" href="https://genius.com/albums/Missy-elliott/The-cookbook" title="The Cookbook">
    <img alt="Https%3a%2f%2fimages" src="https://t2.genius.com/unsafe/64x64/https%3A%2F%2Fimages.genius.com%2F49c23ba7107cf9bb93749df01b30111c.1000x1000x1.jpg" srcset="https://t2.genius.com/unsafe/128x128/https%3A%2F%2Fimages.genius.com%2F49c23ba7107cf9bb93749df01b30111c.1000x1000x1.jpg 2x"/>
    </a>
    <div class="song_album-info">
    <a class="song_album-info-title" href="https://genius.com/albums/Missy-elliott/The-cookbook" title="The Cookbook">
          The Cookbook
          
        </a>
    <a class="song_album-info-artist" href="https://genius.com/artists/Missy-elliott" title="The Cookbook">Missy Elliott</a>
    </div>
    </div>
    <div class="track_listing track_listing--columns">
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">1.  </span>
    <a href="https://genius.com/Missy-elliott-joy-lyrics" title="Joy">
                  Joy
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">2.  </span>
    <a href="https://genius.com/Missy-elliott-partytime-lyrics" title="Partytime">
                  Partytime
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">3.  </span>
    <a href="https://genius.com/Missy-elliott-irresistible-delicious-lyrics" title="Irresistible Delicious">
                  Irresistible Delicious
                  
                </a>
    </div>
    </div>
    <div>
    <span class="track_listing-track track_listing-track--current">
    <span class="track_listing-track_number">4.  </span>
              
              Lose Control
              
              
            </span>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">5.  </span>
    <a href="https://genius.com/Missy-elliott-my-struggles-lyrics" title="My Struggles">
                  My Struggles
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">6.  </span>
    <a href="https://genius.com/Missy-elliott-meltdown-lyrics" title="Meltdown">
                  Meltdown
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">7.  </span>
    <a href="https://genius.com/Missy-elliott-on-and-on-lyrics" title="On &amp; On">
                  On &amp; On
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">8.  </span>
    <a href="https://genius.com/Missy-elliott-we-run-this-lyrics" title="We Run This">
                  We Run This
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">9.  </span>
    <a href="https://genius.com/Missy-elliott-remember-when-lyrics" title="Remember When">
                  Remember When
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">10.  </span>
    <a href="https://genius.com/Missy-elliott-4-my-man-lyrics" title="4 My Man">
                  4 My Man
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">11.  </span>
    <a href="https://genius.com/Missy-elliott-cant-stop-lyrics" title="Can't Stop">
                  Can't Stop
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">12.  </span>
    <a href="https://genius.com/Missy-elliott-teary-eyed-lyrics" title="Teary Eyed">
                  Teary Eyed
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">13.  </span>
    <a href="https://genius.com/Missy-elliott-mommy-lyrics" title="Mommy">
                  Mommy
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">14.  </span>
    <a href="https://genius.com/Missy-elliott-click-clack-lyrics" title="Click Clack">
                  Click Clack
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">15.  </span>
    <a href="https://genius.com/Missy-elliott-time-and-time-again-lyrics" title="Time and Time Again">
                  Time and Time Again
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <span class="track_listing-track_number">16.  </span>
    <a href="https://genius.com/Missy-elliott-bad-man-lyrics" title="Bad Man">
                  Bad Man
                  
                </a>
    </div>
    </div>
    <div>
    <div class="track_listing-track">
    <a href="https://genius.com/Missy-elliott-the-cookbook-tracklist-lyrics" title="The Cookbook [Tracklist]">
                  The Cookbook [Tracklist]
                  
                </a>
    </div>
    </div>
    </div>
    </div>
    </div>
    </div>
    </div>
    </div>
    <ul class="breadcrumbs">
    <li class="breadcrumb " itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb">
    <a href="https://genius.com/" itemprop="url">
    <span itemprop="title">Home</span>
    </a>
    </li>
    <li class="breadcrumb " itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb">
    <a href="https://genius.com/artists-index/m" itemprop="url">
    <span itemprop="title">M</span>
    </a>
    </li>
    <li class="breadcrumb " itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb">
    <a href="https://genius.com/artists/Missy-elliott" itemprop="url">
    <span itemprop="title">Missy Elliott</span>
    </a>
    </li>
    <li class="breadcrumb breadcrumb-current_page" itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb">
    <a href="https://genius.com/Missy-elliott-lose-control-lyrics" itemprop="url">
    <span itemprop="title">Lose Control Lyrics</span>
    </a>
    </li>
    </ul>
    </ng-non-bindable>
    </routable-page>
    <div class="page_footer page_footer--padding-for-sticky-player">
    <div class="footer">
    <div>
    <a href="/about">About Genius</a>
    <a href="/contributor_guidelines">Contributor Guidelines</a>
    <a href="/press" target="_blank">Press</a>
    <a href="mailto:inquiry@genius.com">Advertise</a>
    <a href="https://eventspace.genius.com/">Event Space</a>
    </div>
    <div>
    <a href="/static/privacy_policy" rel="nofollow" target="_blank">Privacy Policy</a>
    <a href="/static/licensing" rel="nofollow" target="_blank">Licensing</a>
    <a href="/jobs">Jobs</a>
    <a href="/developers">Developers</a>
    <a href="/static/terms" rel="nofollow" target="_blank">Terms of Use</a>
    <a href="/static/copyright" rel="nofollow" target="_blank">Copyright Policy</a>
    <a href="/feedback/new" ng-if="::false" rel="nofollow">Contact us</a>
    <contact-button></contact-button>
    <a class="facebox" href="/login" rel="nofollow">Sign in</a>
    </div>
    <div>
    <span class="footer-copyright">© 2018 Genius Media Group Inc.</span>
    </div>
    </div>
    <div class="footer footer--secondary">
    <a class="footer-artist_links_label" href="/verified-artists">Verified Artists</a>
    <span class="footer-artist_links_label">All Artists:</span>
    <ul class="characters_index_list">
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/a">A</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/b">B</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/c">C</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/d">D</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/e">E</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/f">F</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/g">G</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/h">H</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/i">I</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/j">J</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/k">K</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/l">L</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/m">M</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/n">N</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/o">O</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/p">P</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/q">Q</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/r">R</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/s">S</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/t">T</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/u">U</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/v">V</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/w">W</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/x">X</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/y">Y</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/z">Z</a>
    </li>
    <li class="character_index_list-element">
    <a class="character_index_list-link" href="https://genius.com/artists-index/0">#</a>
    </li>
    </ul>
    </div>
    </div>
    <img height="0" src="https://loadus.exelator.com/load/?g=4&amp;j=0&amp;p=1183&amp;page-album_ids=%5B6618%5D%2C%5B335070%5D&amp;page-album_in_top_10=false&amp;page-albums=The+Cookbook%2CRespect+M.E.&amp;page-artist_ids=%5B1529%5D%2C%5B1630%5D%2C%5B3516%5D&amp;page-artist_in_top_10=false&amp;page-artists=Missy+Elliott%2CCiara%2CFatman+Scoop&amp;page-genre_ids=%5B1434%5D&amp;page-genres=Rap+Genius&amp;page-in_top_10=false&amp;page-new_release=false&amp;page-release_decade=2000&amp;page-release_month=200505&amp;page-release_year=2005&amp;page-type=song&amp;user_signed_in=false" style="display: block;" width="0"/>
    <script type="text/javascript">_qevents.push({ qacct: "p-f3CPQ6vHckedE"});</script>
    <noscript>
    <div style="display: none;">
    <img alt="Quantcast" height="1" src="http://pixel.quantserve.com/pixel/p-f3CPQ6vHckedE.gif" width="1">
    </img></div>
    </noscript>
    <script type="text/javascript">
    
      var _sf_async_config={};
    
      _sf_async_config.uid = 3877;
      _sf_async_config.domain = 'genius.com';
      _sf_async_config.title = 'Missy Elliott – Lose Control Lyrics | Genius Lyrics';
      _sf_async_config.sections = 'songs,tag:rap';
      _sf_async_config.authors = 'Missy Elliott,Ciara,Fatman Scoop';
    
      var _cbq = window._cbq || [];
    
      (function(){
        function loadChartbeat() {
          window._sf_endpt=(new Date()).getTime();
          var e = document.createElement('script');
          e.setAttribute('language', 'javascript');
          e.setAttribute('type', 'text/javascript');
          e.setAttribute('src',
             (("https:" == document.location.protocol) ? "https://s3.amazonaws.com/" : "http://") +
             "static.chartbeat.com/js/chartbeat.js");
          document.body.appendChild(e);
        }
        var oldonload = window.onload;
        window.onload = (typeof window.onload != 'function') ?
           loadChartbeat : function() { oldonload(); loadChartbeat(); };
      })();
    </script>
    <!-- Begin comScore Tag -->
    <script>
      var _comscore = _comscore || [];
      _comscore.push({ c1: "2", c2: "17151659" });
      (function() {
        var s = document.createElement("script"), el = document.getElementsByTagName("script")[0]; s.async = true;
        s.src = (document.location.protocol == "https:" ? "https://sb" : "http://b") + ".scorecardresearch.com/beacon.js";
        el.parentNode.insertBefore(s, el);
      })();
    </script>
    <noscript>
    <img src="http://b.scorecardresearch.com/p?c1=2&amp;c2=17151659&amp;cv=2.0&amp;cj=1"/>
    </noscript>
    <!-- End comScore Tag -->
    <script type="text/javascript">(function(e,b){if(!b.__SV){var a,f,i,g;window.mixpanel=b;a=e.createElement("script");a.type="text/javascript";a.async=!0;a.src=("https:"===e.location.protocol?"https:":"http:")+'//cdn.mxpnl.com/libs/mixpanel-2.2.min.js';f=e.getElementsByTagName("script")[0];f.parentNode.insertBefore(a,f);b._i=[];b.init=function(a,e,d){function f(b,h){var a=h.split(".");2==a.length&&(b=b[a[0]],h=a[1]);b[h]=function(){b.push([h].concat(Array.prototype.slice.call(arguments,0)))}}var c=b;"undefined"!==
    typeof d?c=b[d]=[]:d="mixpanel";c.people=c.people||[];c.toString=function(b){var a="mixpanel";"mixpanel"!==d&&(a+="."+d);b||(a+=" (stub)");return a};c.people.toString=function(){return c.toString(1)+".people (stub)"};i="disable track track_pageview track_links track_forms register register_once alias unregister identify name_tag set_config people.set people.set_once people.increment people.append people.track_charge people.clear_charges people.delete_user".split(" ");for(g=0;g<i.length;g++)f(c,i[g]);
    b._i.push([a,e,d])};b.__SV=1.2}})(document,window.mixpanel||[]);
    mixpanel.init('77967c52dc38186cc1aadebdd19e2a82');</script>
    </body>
    </html>



- Using Genius API, this information is more easily accessible.



```python
requests.get('https://api.genius.com', data={'q': 'Lose Control Missy Elliott'}, headers={'Authorization': 'Bearer ' + Genius_TOKEN}).text
```





    '{"meta":{"status":403,"message":"Action forbidden for current scope"}}'



### (4) Lyrics Wiki

- Lyrics Wiki (http://lyrics.wikia.com/wiki/LyricWiki) is a large database for lyrics. Lyrics for a certain song can be scraped at http://lyrics.wikia.com/wiki/[artist_name]:[track_name].


- Below shows a screen capture of http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control, which is the 0th track in the 0th playlist in the 0th file, along with the developer tool.
<img src="fig\LyricsWiki.png" width="635" height="460">


- Below shows a html-parsed result of this page by using `requests` and `BeautifulSoup`:



```python
LyricsWikiURL = "http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control"
BeautifulSoup(requests.get(LyricsWikiURL).text,'html.parser')
```





    <!DOCTYPE doctype html>
    
    <html class="" dir="ltr" lang="en">
    <head>
    <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
    <meta content="width=device-width, user-scalable=yes" name="viewport"/>
    <meta content="MediaWiki 1.19.24" name="generator">
    <meta content="Lose Control lyrics,Missy Elliott Lose Control lyrics,Lose Control by Missy Elliott lyrics,lyrics,LyricWiki,LyricWiki,lyricwiki,Missy Elliott:Lose Control,Missy Elliott,Missy Elliott:The Cookbook (2005),Missy Elliott:Lose Control,Language/English" name="keywords">
    <meta content="Lose Control This song is by Missy Elliott and appears on the album The Cookbook (2005). (Hot Streak - Body Work Sample) Music make you lose control, Music make you lose control. (Fatman Scoop) Let&amp;amp;#39;s go. Hey, yeah, yeah, yeah, yeah, yeah. Here we go now, Here we go now, Here we go now, Here..." name="description"/>
    <meta content="summary" name="twitter:card"/>
    <meta content="@getfandom" name="twitter:site"/>
    <meta content="http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control" name="twitter:url"/>
    <meta content="Missy Elliott:Lose Control Lyrics | LyricWiki | FANDOM powered by..." name="twitter:title"/>
    <meta content="Lose Control This song is by Missy Elliott and appears on the album The Cookbook (2005). (Hot Streak - Body Work Sample) Music make you lose control, Music make you lose control. (Fatman Scoop..." name="twitter:description"/>
    <link href="http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control" rel="canonical"/>
    <link href="/wiki/Missy_Elliott:Lose_Control?action=edit" rel="alternate" title="Edit" type="application/x-wiki"/>
    <link href="/wiki/Missy_Elliott:Lose_Control?action=edit" rel="edit" title="Edit"/>
    <link href="https://vignette.wikia.nocookie.net/lyricwiki/images/b/bc/Wiki.png/revision/latest?cb=20161116095817" rel="apple-touch-icon" sizes="135x135"/>
    <link href="https://vignette.wikia.nocookie.net/lyricwiki/images/6/64/Favicon.ico/revision/latest?cb=20161120182255" rel="shortcut icon"/>
    <link href="/opensearch_desc.php" rel="search" title="LyricWiki (en)" type="application/opensearchdescription+xml"/>
    <link href="http://lyrics.wikia.com/api.php?action=rsd" rel="EditURI" type="application/rsd+xml"/>
    <link href="/wiki/LyricWiki:Copyrights" rel="copyright"/>
    <link href="/wiki/Special:RecentChanges?feed=atom" rel="alternate" title="LyricWiki Atom feed" type="application/atom+xml"/>
    <title>Missy Elliott:Lose Control Lyrics | LyricWiki | FANDOM powered by Wikia</title>
    <!-- CSS injected by skin and extensions -->
    <link href="https://slot1-images.wikia.nocookie.net/__am/7800027800012/sasses/background-dynamic%3Dfalse%26background-image%3Dhttps%253A%252F%252Fvignette3.wikia.nocookie.net%252Flyricwiki%252Fimages%252F5%252F50%252FWiki-background%252Frevision%252Flatest%253Fcb%253D20140425192856%26background-image-height%3D1080%26background-image-width%3D2100%26color-body%3D%2523380759%26color-body-middle%3D%2523fff%26color-buttons%3D%2523006cb0%26color-community-header%3D%2523380759%26color-header%3D%25233a5766%26color-links%3D%2523006cb0%26color-page%3D%2523ffffff%26oasisTypography%3D1%26page-opacity%3D100%26widthType%3D0/skins/oasis/css/oasis.scss,extensions/wikia/DesignSystem/styles/design-system.scss,extensions/wikia/CommunityHeader/styles/index.scss,extensions/wikia/PageHeader/styles/index.scss,extensions/wikia/Recirculation/styles/recirculation.scss,extensions/wikia/PortableInfobox/styles/PortableInfobox.scss,extensions/wikia/PortableInfobox/styles/PortableInfoboxEuropaTheme.scss,extensions/wikia/Qualaroo/css/Qualaroo.scss" rel="stylesheet"/><link href="/load.php?cb=7800027800012&amp;debug=false&amp;lang=en&amp;modules=site&amp;only=styles&amp;skin=oasis&amp;*" rel="stylesheet"/><style>a:lang(ar),a:lang(ckb),a:lang(fa),a:lang(kk-arab),a:lang(mzn),a:lang(ps),a:lang(ur){text-decoration:none}a.new,#quickbar a.new{color:#ba0000}
    
    /* cache key: lyricwiki:resourceloader:filter:minify-css:7:c88e2bcd56513749bec09a7e29cb3ffa */</style><style type="text/css">/*<![CDATA[*/
    .lyricbox
    {
    	padding: 1em;
    	border: 1px solid #ccc;
    	color: #3a3a3a;
    	background-color: #f8f8f8;
    }
    /*]]>*/</style><style type="text/css">/*<![CDATA[*/
    .lyricbox
    {
    	padding: 1em;
    	border: 1px solid #ccc;
    	color: #3a3a3a;
    	background-color: #f8f8f8;
    }
    .lyricsbreak{
    	clear:both;
    }
    /*]]>*/</style>
    <script>
    var Wikia={},
    wgUseSiteJs=true,
    wgWikiVertical="music",
    wgWikiCategories=["music"],
    wgMessages={"categoryselect-button-save":"Save","categoryselect-category-add":"Add category...","categoryselect-category-edit":"Edit category","categoryselect-category-remove":"Remove category","categoryselect-error-category-name-length":"The maximum length for a category name has been reached.","categoryselect-error-duplicate-category-name":"Category \"$1\" already exists.","categoryselect-error-empty-category-name":"Please provide a category name.","categoryselect-modal-category-name":"Provide the name of the category:","categoryselect-modal-category-sortkey":"Optionally, you may alphabetize this page on the \"$1\" category page under the name:","categoryselect-tooltip-add":"Press the Enter or Return key when done."},
    wgOnSiteNotificationsApiUrl="https://services.wikia.com/on-site-notifications",
    JSSnippetsStack=[],
    ads={"context":{"opts":{"adsInContent":1,"delayBtf":true,"enableAdsInMaps":true,"pageType":"all_ads","showAds":true},"targeting":{"enableKruxTargeting":true,"enablePageCategories":true,"esrbRating":"teen","mappedVerticalName":"ent","pageArticleId":275575,"pageIsArticle":true,"pageName":"Missy_Elliott:Lose_Control","pageType":"article","skin":"oasis","wikiCategory":"music","wikiCustomKeyValues":"age=under18;age=13-17;age=18-24;age=kids;age=teen;media=music","wikiDbName":"lyricwiki","wikiId":"43339","wikiIsTop1000":true,"wikiLanguage":"en","wikiVertical":"music","newWikiCategories":["music"]},"providers":{"audienceNetwork":true},"slots":{"invisibleHighImpact":true},"forcedProvider":null},"runtime":{"disableBtf":false}},
    adslots2=[],
    adDriver2ForcedStatus=[],
    wgGaHasAds=true,
    wgAfterContentAndJS=[],
    wgCdnRootUrl="https://slot1-images.wikia.nocookie.net",
    wgCdnApiUrl="https://api.wikia.nocookie.net/__cb7800027800012",
    wgDBname="lyricwiki",
    wgCityId="43339",
    wgContentLanguage="en",
    wgUserName=null,
    wgArticleId=275575,
    wgCategories=["Song","Green Songs","Language/English","Songs by Missy Elliott","Songs L","ITunes/Song","Spotify/Song","Allmusic/Song","MusicBrainz/Song"],
    wgPageName="Missy_Elliott:Lose_Control",
    wikiaPageType="article",
    wikiaPageIsCorporate=false,
    wgArticleType="",
    wgNamespaceNumber=0,
    skin="oasis",
    _gaq=[],
    wgIsGASpecialWiki=true,
    wgStyleVersion="7800027800012",
    wgTransactionContext={"type":"page/main/view/oasis/no_parser/average","env":"prod","php_version":"7.0.27","wiki":"lyricwiki","entry_point":"page","namespace":0,"logged_in":false,"action":"view","skin":"oasis","parser_cache_used":true,"size_category":"average"},
    wgDiscussionsApiUrl="https://services.wikia.com/discussion",
    wgCookieDomain=".wikia.com",
    wgCookiePath="/",
    wgPassiveAutologinUrl="https://services.fandom.com/autologin/passive_frame",
    wgTrustedAutologinUrl="https://services.fandom.com";
    </script><script>window.mw||(mw={fk:1,loader:{state:function(s){preMwLdrStA=window.preMwLdrStA||[];preMwLdrStA.push(s)}}});</script>
    <script>var wgNow = new Date();</script>
    <script>Wikia.InstantGlobals={"wgAdDriverAbTestIdTargeting":80,"wgAdDriverAdditionalVastSizeCountries":["CH"],"wgAdDriverAdEngine3Countries":["XX","test"],"wgAdDriverAolBidderCountries":["XX"],"wgAdDriverAolOneMobileBidderCountries":["XX"],"wgAdDriverAppNexusAstBidderCountries":["XX"],"wgAdDriverAppNexusBidderCountries":["XX"],"wgAdDriverAppNexusDfpCountries":["XX"],"wgAdDriverAudienceNetworkBidderCountries":["XX"],"wgAdDriverA9BidderCountries":["XX"],"wgAdDriverA9DealsCountries":["XX"],"wgAdDriverA9OptOutCountries":["PL","XX-EU"],"wgAdDriverA9VideoBidderCountries":["XX"],"wgAdDriverBabDetectionDesktopCountries":["XX"],"wgAdDriverBabDetectionMobileCountries":["XX"],"wgAdDriverBeachfrontBidderCountries":["XX"],"wgAdDriverBfabStickinessOasisCountries":["XX\/10"],"wgAdDriverBillTheLizardConfig":{"projects":{"queen_of_hearts":[{"name":"ctp_desktop:2.0.0","countries":["XX\/0.05-cached"],"on_0":["disableAutoPlay"],"on_1":["disableAutoPlay"]},{"name":"queen_of_hearts","countries":["XX\/0.05"],
    "dfp_targeting":!0,"on_1":["disableAutoPlay"]}],"cheshirecat":[{"name":"cheshirecat","countries":["PL\/100","NG\/100","MA\/100","UK\/100","GB\/100"]}],"vcr":[{"name":"vcr","dfp_targeting":!0,"countries":["PL\/100","CZ\/100","HK\/100"]}]},"timeout":2000},"wgAdDriverBottomLeaderBoardLazyPrebidCountries":["XX\/50"],"wgAdDriverBottomLeaderBoardAdditionalSizesCountries":["XX"],"wgAdDriverBlockDelayServicesCountries":["disabled"],"wgAdDriverCTPDesktopRabbitCountries":["XX\/0.05"],"wgAdDriverCTPMobileRabbitCountries":["XX\/0.05"],"wgAdDriverCTPDesktopQueenCountries":["XX\/1"],"wgAdDriverDelayCountries":["XX"],"wgAdDriverDelayTimeout":2000,"wgAdDriverDisableSraCountries":["XX"],"wgAdDriverFVAsUapKeyValueCountries":["TJK"],"wgAdDriverFVDelayCountries":["XX"],"wgAdDriverFVDelayTimeoutOasis":2000,"wgAdDriverFVDelayTimeoutMobileWiki":2000,"wgAdDriverHighImpactSlotCountries":["XX"],"wgAdDriverHighImpact2SlotCountries":["XX"],"wgAdDriverIncontentPlayerRailCountries":["NG"],
    "wgAdDriverIncontentPlayerSlotCountries":["XX"],"wgAdDriverIndexExchangeBidderCountries":["XX"],"wgAdDriverKargoBidderCountries":["US"],"wgAdDriverKikimoraPlayerTrackingCountries":["XX"],"wgAdDriverKikimoraTrackingCountries":["XX"],"wgAdDriverKikimoraViewabilityTrackingCountries":["XX"],"wgAdDriverGeoEdgeCountries":["IL","PL"],"wgAdDriverKruxCountries":["AU","NZ","BR","CA","CL","FR","DE","IT","MX","ES","GB","US","PL","JP","UK","SG","RU","SE","BE","NL","DK","NO","FI"],"wgAdDriverKruxNewParamsCountries":["XX"],"wgAdDriverKILOCountries":["XX"],"wgAdDriverLABradorDfpKeyvals":["wgAdDriverCTPDesktopQueenCountries_A_99:queen_ctp_a","wgAdDriverCTPDesktopQueenCountries_B_1:queen_ctp_b","wgAdDriverBfabStickinessOasisCountries_A_90:oasis_stickyblb_a","wgAdDriverBfabStickinessOasisCountries_B_10:oasis_stickyblb_b","wgAdDriverF2BfabStickinessCountries_A_99:ns_sticky_blb_a","wgAdDriverF2BfabStickinessCountries_B_1:ns_sticky_blb_b"],"wgAdDriverLABradorTestCountries":["PL\/40-cached"],
    "wgAdDriverMEGACountries":["XX"],"wgAdDriverMegaAdUnitBuilderForFVCountries":["XX"],"wgAdDriverMoatTrackingForFeaturedVideoAdCountries":["US","PL","AU","DE","UK","GB"],"wgAdDriverMoatTrackingForFeaturedVideoAdSampling":100,"wgAdDriverMoatYieldIntelligenceCountries":["ID","GB","UK"],"wgAdDriverMobileNivensRabbitCountries":["XX"],"wgAdDriverMobileSectionsCollapseCountries":["XX","non-GT"],"wgAdDriverN1DecisionTreeClassifierRabbitCountries":["disable"],"wgAdDriverN1LogisticRegressionRabbitCountries":["disable"],"wgAdDriverNetzAthletenCountries":["DE"],"wgAdDriverOpenXPrebidBidderCountries":["XX"],"wgAdDriverPlayAdsOnNextFVCountries":["XX"],"wgAdDriverPlayAdsOnNextFVFrequency":1,"wgAdDriverPorvataMoatTrackingCountries":["US","PL","AU","DE","UK","GB","MY","CA"],"wgAdDriverPorvataMoatTrackingSampling":25,"wgAdDriverPrebidAdEngine3Countries":["XX"],"wgAdDriverPrebidBidderCountries":["XX"],"wgAdDriverPrebidOptOutCountries":["PL","XX-EU"],"wgAdDriverPubMaticBidderCountries":["XX"],
    "wgAdDriverRabbitTargetingKeyValues":["mnivens","queendesktop","ctpmobile","ctpdesktop"],"wgAdDriverRepeatMobileIncontentCountries":["XX"],"wgAdDriverRubiconDisplayPrebidCountries":["XX"],"wgAdDriverRubiconPrebidCountries":["XX"],"wgAdDriverRubiconDfpCountries":["XX"],"wgAdDriverRubiconVideoInFeaturedVideoCountries":["XX"],"wgAdDriverScrollDepthTrackingCountries":["PL"],"wgAdDriverSrcPremiumCountries":["XX","non-CA"],"wgAdDriverStickySlotsLines":["4883434223","4883432831","4883278922","4864769898","4861785972","4861785324","4837015184","4836650534","4835850889","4833305465","4833302594","4832595783","4831844156","4823785120","4794674861","4852152161","4496549982:UK\/50","4407090664:UK\/50","4387350711:UK\/50","338403012:UK\/50","335687892:UK\/50","4496549982:GB\/50","4407090664:GB\/50","4387350711:GB\/50","338403012:GB\/50","335687892:GB\/50","4745459517","4745458971"],"wgAdDriverWadBTCountries":["XX"],"wgAdDriverWadHMDCountries":["PL-WP","DE-BE"],"wgEnableTrackingOptInModal":!0,
    "wgEnableCMPCountries":["XX"],"wgMobileQualaroo":!0,"wgArticleVideoAutoplayCountries":["XX"],"wgArticleVideoNextVideoAutoplayCountries":["XX"]};</script>
    <script src="https://slot1-images.wikia.nocookie.net/__load/-/cb%3D7800027800012%26debug%3Dfalse%26lang%3Den%26only%3Dscripts%26skin%3Doasis/wikia.ext.instantGlobals,instantGlobalsOverride,abt3sting"></script>
    <script src="https://slot1-images.wikia.nocookie.net/__load/-/cb%3D7800027800012%26debug%3Dfalse%26lang%3Den%26only%3Dscripts%26skin%3Doasis/amd|wikia.tracker.stub,stub|wikia.abTest,cache,cookies,document,geo,instantGlobals,location,log,querystring,window"></script>
    <script>/*<![CDATA[*/window.mw.fk&&delete mw;/*]]>*/</script>
    <!-- Wikia Beacon Tracking -->
    <script>
    	(function () {
    		function genUID() {
    			return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    				var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    				return v.toString(16);
    			});
    		}
    
    		function getCookieValue(cookieName) {
    			var cookieSplit = ('; ' + document.cookie).split('; ' + cookieName + '=');
    
    			return cookieSplit.length === 2 ? cookieSplit.pop().split(';').shift() : null;
    		}
    
    		var expireDate = new Date(),
    			beacon = getCookieValue('wikia_beacon_id'),
    			sessionId = getCookieValue('tracking_session_id'),
    			pvNumber = getCookieValue('pv_number'),
    			pvNumberGlobal = getCookieValue('pv_number_global');
    
    		if (beacon) {
    			window.beacon_id = beacon;
    		}
    
    		window.sessionId = sessionId ? sessionId : genUID();
    		window.pvNumber = pvNumber ? parseInt(pvNumber, 10) + 1 : 1;
    		window.pvNumberGlobal = pvNumberGlobal ? parseInt(pvNumberGlobal, 10) + 1 : 1;
    		window.pvUID = genUID();
    
    		expireDate = new Date(expireDate.getTime() + 1000 * 60 * 30);
    		document.cookie = 'tracking_session_id=' + window.sessionId + '; expires=' + expireDate.toGMTString() +
    			';domain=' + window.wgCookieDomain + '; path=' + window.wgCookiePath + ';';
    		document.cookie = 'pv_number=' + window.pvNumber + '; expires=' + expireDate.toGMTString() +
    			'; path=' + window.wgCookiePath + ';';
    		document.cookie = 'pv_number_global=' + window.pvNumberGlobal + '; expires=' + expireDate.toGMTString() +
    			';domain=' + window.wgCookieDomain + '; path=' + window.wgCookiePath + ';';
    	})();
    </script>
    <script src="https://slot1-images.wikia.nocookie.net/__am/7800027800012/groups/-/abtesting,oasis_blocking,universal_analytics_js,adengine2_top_js,adengine2_a9_js,adengine2_pr3b1d_js,tracking_opt_in_js,qualaroo_blocking_js" type="text/javascript"></script>
    <!-- Make IE recognize HTML5 tags. -->
    <!--[if IE]>
    	<script>/*@cc_on'abbr article aside audio canvas details figcaption figure footer header hgroup mark menu meter nav output progress section summary time video'.replace(/\w+/g,function(n){document.createElement(n)})@*/</script>
    <![endif]-->
    <meta content="112328095453510" prefix="fb: http://www.facebook.com/2008/fbml" property="fb:app_id"/>
    <meta content="article" property="og:type"/>
    <meta content="LyricWiki" property="og:site_name"/>
    <meta content="Missy Elliott:Lose Control" property="og:title"/>
    <meta content="Lose Control This song is by Missy Elliott and appears on the album The Cookbook (2005). (Hot Streak - Body Work Sample) Music make you lose control, Music make you lose control. (Fatman Scoop) Let&amp;#39;s go. Hey, yeah, yeah, yeah, yeah, yeah. Here we go now, Here we go now, Here we go now, Here we go now. One time now. Misdemeanor&amp;#39;s in the house. Ciara&amp;#39;s in the house. Misdemeanor&amp;#39;s in the house. Fatman Scoop, Man Scoop, Man Scoop. (Verse 1: Missy Elliott) I&amp;#39;ve got a cute..." property="og:description"/>
    <meta content="http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control" property="og:url"/>
    <meta content="app-id=447519370, app-arguments=http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control" name="apple-itunes-app"/>
    </meta></meta></head>
    <body class="mediawiki ltr ns-0 ns-subject page-Missy_Elliott_Lose_Control oasis-breakpoints skin-oasis user-anon background-fixed wiki-lyricwiki">
    <script type="text/javascript">
    	function runILCode() {
    		//Copyright Instart Logic Fri Nov 02 2018 - All rights reserved - version: 10.3.20
    !function(t){if(function(){var n=!0,i=!1,e="";try{var r="object"==typeof t.IXC_306_6624720851103629&&t.IXC_306_6624720851103629;if(n=!r||void 0===r.CanRun||"_306_6624720851103629"!==r._306_6624720851103629||r.CanRun("abd"))r=t.IXC_306_6624720851103629=t.IXC_306_6624720851103629||{},r.InitStartTime=(t.performance?t.performance:Date).now(),r._306_6624720851103629="_306_6624720851103629",t.INSTART_TARGET_NAME="abd",t.I11C=t.I11C||{};else{var o=t.INSTART_TARGET_NAME;e="double nanovisor injection: abd after "+o,(i="abd"!==o)&&console.warn(e)}t.INSTART=t.INSTART||{},t.INSTART.Init=function(){i&&console.error("ignored Init call after "+e)}}catch(t){try{console.error(t)}catch(t){}n=!0}return n}()){!function(){function n(n,i){function e(i){return n.req.a(t.atob(i))}(e("bWVkaWNpbmVuZXQuY29t")||e("b25oZWFsdGguY29t")||e("cnhsaXN0LmNvbQ==")||e("d2VibWQuY29t"))&&n.req.b("firefox")&&i({iadb:{ierelbanenoitcerid:!0}})}function i(){return(i=Object.assign||function(t){for(var n,i=1,e=arguments.length;i<e;i++){n=arguments[i];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(t[r]=n[r])}return t}).apply(this,arguments)}function e(t,n){return(e=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(t,n){t.__proto__=n}||function(t,n){for(var i in n)n.hasOwnProperty(i)&&(t[i]=n[i])})(t,n)}function r(){if(void 0!==z)return z;var n=!0;try{if("l2n9s8hg5p"!==t.atob(t.btoa("l2n9s8hg5p")))throw"l2n9s8hg5p";n=!1}catch(t){}return z=n}function o(t){return void 0===t&&(t=I),1===t.v.iytep}function u(t){return void 0===t&&(t=I),4===t.v.iytep}function c(t){return void 0===t&&(t=I),2===t.v.iytep}function s(t,n){function i(){this.constructor=t}e(t,n),t.prototype=null===n?Object.create(n):(i.prototype=n.prototype,new i)}function a(t,n){t||(console.error(n),0<(t=[]).length&&console.log(t))}function h(t){return t.split("").filter(function(t,n){return 0==n%2}).reverse().join("")}function d(){if(void 0!==At)return At;var t=!1,n=Object.getOwnPropertyDescriptor(HTMLFrameElement.prototype,"contentWindow");return(n=n&&n.get&&n.get.toString())&&(t=-1===n.indexOf(K)),At=t}function l(t){var n=Xt[t]?Xt[t]:null;if(n)return m(n);if(!(n=zt.exec(""+t)))return{ihcseme:null,iedercslaitn:null,imodnia:null,ioptr:null,ihtapeman:t,iuqyre:null,igarftnem:null};var i=m(n={ihcseme:f(n[1]),iedercslaitn:f(n[2]),imodnia:f(n[3]),ioptr:f(n[4]),ihtapeman:f(n[5]),iuqyre:b(n[6]),igarftnem:b(n[7])}),e=Object.keys(Xt);return 0!==e.length&&e.length>Gt-1&&delete Xt[e[e.length%Gt]],Xt[t]=i,n}function f(t){return"string"==typeof t&&0<t.length?t:null}function b(t){return"string"==typeof t?t:null}function m(t){return{ihcseme:t.ihcseme,iedercslaitn:t.iedercslaitn,imodnia:t.imodnia,ioptr:t.ioptr,ihtapeman:t.ihtapeman,iuqyre:t.iuqyre,igarftnem:t.igarftnem}}function v(t){return{ihcseme:t[0],iedercslaitn:t[1],imodnia:t[2],ioptr:t[3],ihtapeman:t[4],iuqyre:t[5],igarftnem:t[6]}}function p(){var n=0,i=[];for(i.push(Et+"="+Date.now()+";max-age=86400"),i.push(St+"=1;max-age=86400"),i.push(Yt+"="+document.referrer+";path=/;max-age=10");n<i.length;n++)document.cookie=i[n];n=new nn(t.location.href),i=en.Ka();for(var e=[],r=0,o=0;o<i.length;o++){var u=i.charAt(o);40<o-r||rn.test(u)?(e.push("%"+("00"+u.charCodeAt(0).toString(16)).slice(-2)),r=o):e.push(u)}i="i10c.encReferrer="+e.join(""),e=_t,(r=n.iaptegemanht())&&("/"!==r[0]&&(e+="/"),e+=r),n.iaptesemanht(e),n.iqddayreu(i),n.ieuqddamarapyr("i10c.ua",P.iytep.toString()),n=n.itegferh(),t.location.replace(n)}function y(t,n,i){var e=Object.getOwnPropertyDescriptor(t,n)||{};if(!1!==e.writable){e.writable=!1,e.value=i;try{Object.defineProperty(t,n,e)}catch(t){}}else a(e.value===i,n+" is already defined on object with different value");return i}function L(t,n,i){try{t[n]=i}catch(t){}return i}function R(n,i,e){n=n.split("."),e=e?y:L;for(var r,o=t||{},u=0;u<n.length-1;++u){if(r=n[u],!o)return;o=e(o,r,o[r]||{})}r=n[n.length-1],o&&(a(!!r,"In Export, name not defined"),e(o,r,i))}function g(t,n){R($+"."+t,n),R(J+"."+t,n)}function N(){for(var t=[],n=0;n<arguments.length;n++)t[n]=arguments[n];on?t.forEach(function(t){return t(on)}):un=un.concat(t)}function D(n,i){on=n;var e=!0;try{un.forEach(function(t){e=t(n)&&e});var r=i&&t[i];r&&r instanceof Array&&(r.forEach(function(t){e=t(n)&&e}),r.push=N)}catch(t){e=!1}return e}function w(t,n,i,e){if(t||(t={}),n)for(var r=Object.keys(n),o=r.length-1;0<=o;--o){var u=r[o],c=t[u];c=(void 0===c?sn:e&&e[u]||i)(c,n[u]),t[u]=c}return t}function E(t,n){return function(i,e){return null===e||"object"!=typeof e||e instanceof RegExp?e:Array.isArray(e)?!Array.isArray(i)&&i?(console.error("Error while attempting to merge an array "+e+" with a non-array "+i),e):t(i,e):n(i,e)}}function S(t,n){return w(t,n,sn)}function Y(t,n){for(var i=0;i<n.length;i++)t=w(t,n[i],sn);return t}function _(t,n){var i=n&&Y({},n);return function(n,e){return w(n,e,t,i)}}function j(t,n){return w(t,n,an)}function T(t,n){return w(t,n,hn)}function H(t,n){return t?n?function(){t.apply(this,arguments),n.apply(this,arguments)}:t:n}function W(t,n){return t===n}function F(t,n){return null===t?t===n:t.toLowerCase()===n.toLowerCase()}function Z(t,n){return null!==t&&void 0!==t&&n.test(t)}function M(t,n){for(var i=0;i<t.length;i++)if(t[i]===n)return!0;return!1}function k(t,n){void 0===n&&(n=Cn);var i=new n.Da,e=null;if("string"!=typeof t.url&&(e=t.url.itegferh()),t.url=e||t.url,t.onload&&(i.onload=function(){return t.onload(new An(i,t))}),t.onerror&&(i.onerror=function(){return t.onerror(new An(i,t))}),i.withCredentials=!!t.credentials,t.bb&&i.overrideMimeType(t.bb),n.Sa.call(i,t.method,t.url,!1!==t.async),t.S)for(e=0;e<t.S.length;e++)i.setRequestHeader(t.S[e].ianem,t.S[e].Ya);e=t.body&&("string"==typeof t.body?t.body:JSON.stringify(t.body));var r=!1===t.async;try{n.Wa.call(i,e)}catch(n){r&&t.onerror&&t.onerror(null)}return r?new An(i,t):null}function V(t){return t[Tt]||t[Zt]===ht}function O(t,n,i){var e;return e={},e[Zt]=t,e[Tt]=n,t=e,i&&(t[Ht]=i),t}function x(t,n){var i="";if(t){var e=t.indexOf(n);-1<e&&(t=t.slice(e),n=n.replace(/[.]/g,"[.]").replace(/\$/g,"[$]"),(t=new RegExp("(?:\\b"+n+"\\b\\s*=\\s*)([^;]*);?").exec(t))&&t[1]&&(i=t[1]))}return i}function A(){var t={};try{if(C(Vt)){var n=sessionStorage._306_6624720851103629;n&&(t=JSON.parse(n))}}catch(t){}return t}function C(n){try{var i=t[n];return i.setItem("TEST","TEST"),i.removeItem("TEST"),!0}catch(t){return!1}}function B(){return new ii(function(t){function n(n){var i=!1;"load"===n.type?i=!1:"error"===n.type&&(i=!0),t(O(ct,i,n.currentTarget.src))}var i=document.createElement("img");i.onload=n,i.onerror=n;var e=Math.floor(Math.random()*In.length);i.src=In[e]+"?"+ei[Math.floor(Math.random()*ei.length)],1<In.length&&In.splice(e,1)})}function G(t,n,i,e,r){return new ii(function(s){var a="script"===t?ut:st,h=u();if(o()||h||c()){var d;h&&(d=setTimeout(function(){s(O(a,!1,"Timed out on error"))},500));var l=!1,f=function(t){if(!l){l=!0,h&&clearTimeout(d);var n=!1;"load"===t.type?n=!1:"error"===t.type&&(n=!0),s(O(a,n,t.currentTarget.href)),b&&b.parentNode&&b.parentNode.removeChild(b)}},b=document.createElement("link");b.onload=f,b.onerror=f,b.rel=r||c()?"prefetch":"preload",b.as=t,i&&b.setAttribute("crossorigin","anonymous"),e&&b.setAttribute("referrerpolicy","no-referrer"),f=Math.floor(Math.random()*n.length),b.href=n[f],ni&&ni.parentNode?ni&&ni.parentNode&&ni.parentNode.insertBefore(b,ni):document.head.appendChild(b),1<n.length&&n.splice(f,1)}else s(O(a,!1,"Neither Chrome nor Safari"))})}function X(){return new ii(function(t){var n=ri[Math.floor(Math.random()*ri.length)],i=document.createElement(n[0]);i.id=n[1],i.style.width="1px",i.style.height="1px",i.style.top="-555px",i.style.left="-555px",i.style.display="block";var e=setInterval(function(){document&&document.body&&(document.body.appendChild(i),clearInterval(e),setTimeout(function(){var n,e=getComputedStyle(i),r=!1;("none"===e.display||(n=e.getPropertyValue("-moz-binding"))&&-1!==n.indexOf("abp-elemhidehit"))&&(r=!0),t(O(mt,r)),i&&i.parentNode&&i.parentNode.removeChild(i)},200))},10)})}var z=void 0,Q=function(){function t(t){this.v=this.ka(t,[{g:/MSIE\s([\d]+)(\.([\d]+))?(\.([\d]+))?/,iytep:3},{g:/Edge\/([\d]+)(\.([\d]+))?(\.([\d]+))?/,iytep:5},{g:/Trident.*rv:([\d]+)(\.([\d]+))?(\.([\d]+))?/,iytep:3},{g:/Chrome\/([\d]+)(\.([\d]+))?(\.([\d]+))?/,iytep:1},{g:/Firefox\/([\d]+)(\.([\d]+))?(\.([\d]+))?/,iytep:2},{g:/Version\/([\d]+)(\.([\d]+))?(\.([\d]+))?.*Safari/,iytep:4}])||{iytep:0,M:0,X:0,W:0},this.na=this.ka(t,[{g:/Android\s([\d]+)(\.([\d]+))?(\.([\d]+))?/,iytep:1},{g:/CPU OS ([\d]+)(_([\d]+))?(_([\d]+))?/,iytep:2},{g:/CPU iPhone OS ([\d]+)(_([\d]+))?(_([\d]+))?/,iytep:2},{g:/Mac OS X ([\d]+)(_([\d]+))?(_([\d]+))?/,iytep:3},{g:/Windows NT ([\d]+)(.([\d]+))?(.([\d]+))?/,iytep:4},{g:/Linux [a-z]*([\d]+)(_([\d]+))?(_([\d]+))?/,iytep:5},{g:/CrOS [a-z]*([\d]+)(_([\d]+))?(_([\d]+))?/,iytep:5}])||{iytep:0,M:0,X:0,W:0}}return t.prototype.ka=function(t,n){for(var i=0;i<n.length;++i){var e=n[i],r=e.g.exec(t);if(null!==r)return{iytep:e.iytep,M:parseInt(r[1])||0,X:parseInt(r[3])||0,W:parseInt(r[5])||0}}return{iytep:0,M:0,W:0,X:0}},t}(),I=new Q("undefined"!=typeof navigator&&navigator.userAgent?navigator.userAgent:""),P=I.v,U=function(t){function n(){var n=t.call(this)||this;return n.ua=!1,t.prototype.u.call(n,function(){}),n}return s(n,t),n.prototype.L=function(t){return function(){if(!this.ua){for(var n=0,i=t.length;n<i;++n)t[n].apply(this,arguments);this.ua=!0}}},n}(function(t){function n(){var n=t.call(this)||this;return t.prototype.u.call(n,function(){n.Ra=this,n.ma=arguments}),n}return s(n,t),n.prototype.L=function(n){var i=this,e=t.prototype.L.call(this,n);return function(){e.apply(this,arguments),i.Xa()}},n.prototype.u=function(n){this.ma?n.apply(this.Ra,this.ma):t.prototype.u.call(this,n)},n}(function(){function t(){this.i=this.P}return t.prototype.P=function(){},t.prototype.L=function(t){return function(){for(var n=0,i=t.length;n<i;++n)t[n].apply(this,arguments)}},t.prototype.u=function(t){if(this.i===this.P)this.i=t;else if(this.i!==this.ta){var n=this.ea;n||(n=this.ea=[this.i,t],this.i=this.ta=this.L(n))}else this.ea.push(t)},t.prototype.Xa=function(){this.i!==this.P&&(this.i!==this.ta?this.i=this.P:this.ea.length=0)},t}())),J=h("C21719I6"),$=h("C10817I3");h("T0R7A3T4S9N0I9");var q=h("d6e4r8i5f5_0d5a4o2l0n0o0_9e8l9g4o6o4g8"),K=h("e1d2o9c3 8e9v2i3t2a4n6"),tt=h("k7c6a1b9l1l3a8C3d0b4A5r6e5t8s4i7g6e1R9"),nt=h("E0B9U3O4Y2");h("B1U2O5Y7"),h("P4B8A1");var it=h("B0A6");h("S0T5A2T0S2B5D7A2"),h("G3D0A4");var et=h("P7C3X6E7"),rt=h("X4L6O1V2");h("E6N8O7N4");var ot=h("E8V0A7R9B7"),ut=h("T2P9I5R0C7S7B5"),ct=h("G5M6I8B8"),st=h("L6M1T6H8B4"),at=h("I7R6U9D9A6D3E7K7C1O4L7B6"),ht=h("E9M3A3R1F3B0");h("D6E9T2S6E5U3Q2E3R5D4A8P4F2D6");var dt=h("D1E1D6A7O5L5P4F2D7"),lt=h("E9L1B2A2W7E8I3V5P1F7D5"),ft=h("E0G1N4A1H1C3W5E2I5V9P4F2D7"),bt=h("D8E1R2E8D7N5E1R8P8F8D4"),mt=h("T9N2E9M4E2L7E7"),vt=h("D4F2"),pt=h("T4S1I9H6"),yt=h("I2R4U0D2A4D3E7D6A1O6L2"),Lt=h("K1R9O0W5T4E8N4"),Rt=h("O0F3N9I4G5I7S9N9O4N7");h("S1T6A1T9S3D5I9B0E3R4P2"),h("S5T8S6E7T5G1I3S9N3O1N1T1P0R5");var gt=h("T9L2U9S3E0R7D7E6R3O2T5S6"),Nt=h("T0N8E8S2E3R2P4"),Dt=h("T9O3N7"),wt=h("Y2D8A2E4R5T5O3N9"),Et=h("e0t5u3o1r9.6h195r908m9"),St=h("c60914i2h5p0r1o0m2"),Yt=h("r6e6r3r0e3f9e3r0.7c50016i0"),_t=h("0302g3/5");h("e3t1a4t8S2h9p0r8o6M6"),h("d4e7k5c2o7l2b1"),h("d4e1k5c7o1l9b4n7u3");var jt=h("/7)2]5z7-6a790-504[2]493-609[9g8:7?2(5/7^6"),Tt=h("d0e1t1c7e6t5e0D9"),Ht=h("n7o6s6a1e4R2"),Wt=h("s8t4l8u5s4e1R8"),Ft=h("y0c7n7e8t7a2L6n3o1i4t5c7e6t6e9D6"),Zt=h("e7p0y4T8"),Mt=h("D5E3D7A9O4L0_2V1"),kt=h("D5E0D9A8O8L1T6O5N3_9V9"),Vt=h("e6g8a9r1o7t0S5n1o9i2s3s0e8s6"),Ot=h("d0f9c30712i7"),xt=h("l2r6u1d5b7a1t9e8g9.1c80914i3");h("k2c4a0b8l0l6a5c5a7h8c4t7p3a4c7"),h("a5h4c5t3p6a3c8e3r3g8"),h("o1x465N408Y8P8k4d7S9V4k434J1V5v2x11435N468X8q8k4e7v9M4f4Z4z1t5H2"),h("e2i7k6o4o5c1"),h("b9d7d6d5b4.3c20217i3"),h("e9m9o7r8h1c9"),h("i1s4c2"),h("T8d9a2o5l8n4o8"),h("n1o3i4s7s5i3m8r6e5p9"),h("s9n3o6i6s6s3i2m1r9e0p5"),h("y1r8e2u5q7"),h("d5e2i2n2e4d1"),h("t5p4m8o1r6p7"),h("r3e0v7i9r5d9b7e6w2"),h("k1c0i3l7c3"),h("t3r4a0t4s1h9c2u3o9t8"),h("s2s5e6r6p8y0e1k3"),h("e2v2o8m5e9s2u2o9m8"),h("l8l4o5r5c8s3"),h("t3u7p1n3i0"),h("e4t4s0a6p0"),h("n3o2i6t0o7m4e2c1i1v1e6d5"),h("m6o6t4n1a7h6p2_4"),h("m1o5t4n3a7h3P6l3l7a4c7"),h("s1j3m8o0t0n3a1h8p9");var At=void 0,Ct=new U,Bt=o()&&r()&&d();Ct.i(Bt);var Gt=5,Xt={},zt=/^(?:([^:/?#]+):)?(?:\/\/+(?:([^/?#]*)@)?([^/?#:@]*)(?::([0-9]+)?)?)?([^?#]+)?(?:\?([^#]*))?(?:#(.*))?$/,Qt=/[#\/\?@]/g,It=/[#\/\?]/g,Pt=/[#\?]/g,Ut=/[#]/g,Jt=/(\/|^)(?:[^./][^/]*|\.{2,}(?:[^./][^/]*)|\.{3,}[^/]*)\/\.\.(?:\/|$)/,$t=/^([/])?(?:\.\.(?:[/]|$))+/,qt=/(^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?).){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$)|([^.]+[.](?:([^.]{2}[.][^.]{2})|([^.]+))$)/,Kt=/[^.]+[.](no|com|co|io|it|de|net|org|fr|one|tv|today|fm|kr|jp|com[.]au|co[.]uk|ca)$/,tn=/^([^=]*)(?:=(.*))?$/,nn=function(){function t(t){t&&(this.israpirude="string"==typeof t?l(t):t),this.israpirude||(this.israpirude={ihcseme:null,iedercslaitn:null,imodnia:null,ioptr:null,ihtapeman:null,iuqyre:null,igarftnem:null}),this.iacehc={m:null,iironig:null,C:null,G:null,H:null,j:[null]}}return t.prototype.itegferh=function(){var t;if(!(t=this.iacehc.m)){var n="";null!==(t=this.israpirude).ihcseme&&(n+=t.ihcseme+":"),null!==t.imodnia&&(n+="//",null!==t.iedercslaitn&&(n+=t.iedercslaitn+"@"),n+=t.imodnia,null!==t.ioptr&&(n+=":"+t.ioptr)),null!==t.ihtapeman&&(n+=t.ihtapeman),null!==t.iuqyre&&(n+="?"+t.iuqyre),null!==t.igarftnem&&(n+="#"+t.igarftnem),t=this.iacehc.m=n}return t},t.prototype.iotegnigir=function(){var t=this.iacehc;if(!t.iironig){var n="",i=this.istegemehc();this.icurtssilruderut()||!i?(i&&(n+=i+":"),(i=this.itegtsoh())&&(n+="//"+i),t.iironig=""!==n?n:null):t.iironig="null"}return t.iironig},t.prototype.istegemehc=function(){return this.israpirude.ihcseme},t.prototype.iosbasilruetul=function(){return!!this.israpirude.ihcseme&&!!this.israpirude.imodnia},t.prototype.icurtssilruderut=function(){var t=this.israpirude.ihcseme;return null!==t&&("http"===t||"https"===t||"ftp"===t)},t.prototype.istesemehc=function(t){return null!==t&&Qt.test(t)?this:(this.israpirude.ihcseme=t?t.toLowerCase():null,t=this.iacehc,t.m=null,t.iironig=null,this)},t.prototype.irpteglocoto=function(){var t=this.istegemehc();return t&&t+":"},t.prototype.iderctegslaitne=function(){return this.israpirude.iedercslaitn},t.prototype.iderctesslaitne=function(t){return null!==t&&Qt.test(t)?this:(this.israpirude.iedercslaitn=t,this.iacehc.m=null,this)},t.prototype.itegtsoh=function(){var t=this.iacehc;if(!t.C){var n="",i=this.idtegniamo();i&&(n+=i,(i=this.itegtrop())&&(n+=":"+i)),t.C=""!==n?n:null}return t.C},t.prototype.idtegniamo=function(){return this.israpirude.imodnia},t.prototype.idtesniamo=function(t){return null!==t&&It.test(t)?this:(this.israpirude.imodnia=t,(t=this.israpirude.ihtapeman)&&"/"!==t[0]&&(this.israpirude.ihtapeman="/"+t),t=this.iacehc,t.m=null,t.iironig=null,t.C=null,t.G=null,t.H=null,this)},t.prototype.iamodtoorkcehcehcacdltdnani=function(){var t=this.iacehc;if(!t.G){var n=this.idtegniamo();if(n){var i=Kt.exec(n);i?(t.G=i[0],t.H=i[1]):(i=qt.exec(n))&&(t.G=i[0],(t.H=i[4]||null)||(t.H=i[3]))}}},t.prototype.ielpottegniamodlev=function(){return this.iamodtoorkcehcehcacdltdnani(),this.iacehc.H},t.prototype.ioortegniamodt=function(){return this.iamodtoorkcehcehcacdltdnani(),this.iacehc.G||this.idtegniamo()},t.prototype.itegtrop=function(){return this.israpirude.ioptr},t.prototype.itestrop=function(t){if(t){if((t=Number(t))!==(65535&t))return this;this.israpirude.ioptr=""+t}else this.israpirude.ioptr=null;return t=this.iacehc,t.m=null,t.iironig=null,t.C=null,this},t.prototype.iteghtap=function(){var t="",n=this.iaptegemanht();return n&&(t+=n),(n=this.iqtegyreu())&&(t+="?"+n),(n=this.irftegtnemga())&&(t+="#"+n),""!==t?t:null},t.prototype.iaptegemanht=function(){return this.israpirude.ihtapeman},t.prototype.iaptesemanht=function(n,i){if(n){if(Pt.test(n))return this;n=!i||!this.icurtssilruderut()&&this.istegemehc()?n:t.Ca(n),this.israpirude.ihtapeman=this.israpirude.imodnia&&"/"!==n[0]?"/"+n:n||null}else this.israpirude.ihtapeman=null;return this.iacehc.m=null,this},t.prototype.izilamronemanhtape=function(){return this.iaptesemanht(this.israpirude.ihtapeman,!0)},t.prototype.isteghcrae=function(){var t=this.iqtegyreu();return t&&"?"+t},t.prototype.iqtegyreu=function(){return this.israpirude.iuqyre},t.prototype.iqtesyreu=function(t){return null!==t&&Ut.test(t)?this:(this.israpirude.iuqyre=t,t=this.iacehc,t.m=null,t.j=[null],this)},t.prototype.iteghsah=function(){var t=this.irftegtnemga();return t&&"#"+t},t.prototype.irftegtnemga=function(){return this.israpirude.igarftnem},t.prototype.irftestnemga=function(t){return this.israpirude.igarftnem=t,this.iacehc.m=null,this},t.prototype.iqddayreu=function(t){var n=t;return this.israpirude.iuqyre&&(n=this.israpirude.iuqyre+"&"+t),this.iqtesyreu(n)},t.prototype.ieuqddamarapyr=function(t,n){return this.iqddayreu(n?t+"="+n:t)},t.prototype.iarapkcehcehcacretem=function(){var t=this.iacehc;if(1===t.j.length&&null===t.j[0]){var n=this.israpirude.iuqyre;if(n){for(var i=[],e=-1,r=0,o=(n=n.split("&")).length;r<o;++r){var u=tn.exec(n[r]);u&&(i[++e]=u[1],u[2]||""===u[2]?i[++e]=u[2]:i[++e]=null)}t.j=i}else t.j=[]}},t.prototype.iapllatessretemar=function(t){for(var n="",i="",e=t.length,r=0;r<e;){var o=t[r++],u=t[r++];n+=i+o,i="&",(u||""===u)&&(n+="="+u)}return this.iqtesyreu(n||null)},t.prototype.iemaraptesseulavret=function(t,n){"string"==typeof n&&(n=[n]),this.iarapkcehcehcacretem();for(var i,e=0,r=this.iacehc.j,o=[],u=0,c=r.length;u<c;u+=2)t===r[u]?(i=!0,e<n.length&&o.push(t,n[e++])):o.push(r[u],r[u+1]);for(;e<n.length;)o.push(t,n[e++]);return(0<e||i)&&this.iapllatessretemar(o),this},t.prototype.ipevomerretemara=function(t){return this.iemaraptesseulavret(t,[])},t.prototype.iapllategsretemar=function(){return this.iarapkcehcehcacretem(),this.iacehc.j.slice(0,this.iacehc.j.length)},t.prototype.iemaraptegseulavret=function(t){this.iarapkcehcehcacretem();for(var n=this.iacehc.j,i=[],e=0,r=n.length;e<r;e+=2)t===n[e]&&i.push(n[e+1]);return i},t.prototype.imaraptegeulavrete=function(t){this.iarapkcehcehcacretem();for(var n=this.iacehc.j,i=0,e=n.length;i<e;i+=2)if(t===n[i])return n[i+1];return null},t.Ca=function(t){t=t.replace(/(^|\/)\.(?:\/|$)/g,"$1").replace(/\/{2,}/g,"/");for(var n;(n=t.replace(Jt,"$1"))!=t;t=n);return t=t.replace($t,"$1")},t.prototype.itumetaercwodahselba=function(n){return new(n=n||t)(Object.create(this.israpirude))},t.prototype.ierfeze=function(){return Object.freeze(this.israpirude),this},t.prototype.ilceno=function(){return new t({ihcseme:this.israpirude.ihcseme,iedercslaitn:this.israpirude.iedercslaitn,imodnia:this.israpirude.imodnia,ioptr:this.israpirude.ioptr,ihtapeman:this.israpirude.ihtapeman,iuqyre:this.israpirude.iuqyre,igarftnem:this.israpirude.igarftnem})},t.prototype.iiresezila=function(){var t=this.israpirude;return[t.ihcseme,t.iedercslaitn,t.imodnia,t.ioptr,t.ihtapeman,t.iuqyre,t.igarftnem]},t}(),en=null,rn=/[^A-Za-z0-9\-_.]/;!function(t){en||(en=new t)}(function(n){function i(){return null!==n&&n.apply(this,arguments)||this}return s(i,n),i.prototype.Ka=function(){var n=document.referrer;return t.btoa(n)},i}(function(){}));var on,un=[];g(tt,N),r()&&Ct.u(function(t){t&&(t={},t[Wt]=[],t[Ft]=0,t[Tt]=!0,D(t)&&p())});var cn=/webcache[.]googleusercontent[.]com|cc[.]bingj[.]com|web[.]archive[.]org/i,sn=E(function(t,n){return n.slice()},S),an=E(function(t,n){return t.concat(n)},j),hn=E(function(t,n){return t.concat(n.filter(function(n){return 0>t.indexOf(n)}))},T),dn=Object.create(null,{Version:{value:"10.3.20",writable:!1,configurable:!1,enumerable:!0}}),ln=function(t,n,i){var e=this,r=S(i||{},dn);this.l=function(i,e,o,u){e=new t(e);var c=Object.create(r);return o&&(c=n(c,o)),i(e,function(t){return t?n(c,t):c}),u&&(c=n(c,u)),c},this.pa=function(){},this.ComputeAll=function(t,n,i){return e.l(e.pa,t,n,i)}},fn=function(){function t(t){this.s=t instanceof nn?t:new nn(t&&v(t))}return t.prototype.a=function(t){return W(this.s.ioortegniamodt(),t)},t.prototype.scheme=function(t){return W(this.s.istegemehc(),t)},t.prototype.domain=function(t){return Array.isArray(t)?M(t,this.s.idtegniamo()):W(this.s.idtegniamo(),t)},t.prototype.ab=function(t){return Z(this.s.idtegniamo(),t)},t.prototype.cb=function(){return Z(this.s.iqtegyreu(),/abdtrigger/)},t.prototype.R=function(t){return M(this.s.iemaraptegseulavret("abdtrigger"),t)},t.prototype.port=function(t){return W(this.s.itegtrop(),t)},t.prototype.href=function(t){return W(this.s.itegferh(),t)},t}(),bn={ie:3,chrome:1,edge:5,firefox:2,safari:4},mn={android:1,ios:2,mac:3,windows:4,linux:5},vn=function(t){function n(n){(n=n||{}).headers=n.headers||{};var i=n.uri,e=n.url;return(i=i instanceof nn?i:new nn(i?v(i):e?l(e):void 0)).idtegniamo()||(e=new nn("//"+n.headers.host),i.idtesniamo(e.idtegniamo()),i.itestrop(e.itegtrop())),i=t.call(this,i)||this,i.ca=n,n=n.headers["i-resutnega"],i.O=n?new Q(n):I,i}return s(n,t),n.prototype.U=function(t){var n=this.ca.headers;return n&&n[t]||null},n.prototype.ia=function(){var t=this.ca.headers;return t&&t.cookie||""},n.prototype.la=function(t,n,i){return void 0!==(t=t.M)&&(void 0===n||n<=t)&&(void 0===i||t<=i)},n.prototype.method=function(t){return this.ca.method===t},n.prototype.va=function(t,n){return F(this.U(t),n)},n.prototype.host=function(t){return this.va("host",t)},n.prototype.$a=function(t){return Z(this.ia(),t)},n.prototype.cookie=function(t,n){return-1!==this.ia().indexOf(t)&&this.$a(new RegExp(t+"s*=[^;]*"+n+"[^;]*"))},n.prototype.b=function(t,n,i){return void 0===(t=bn[t])&&(t=0),this.O.v.iytep===t&&this.la(this.O.v,n||Number.MIN_VALUE,i||Number.MAX_VALUE)},n.prototype.wa=function(t){return void 0===(t=mn[t])&&(t=0),this.O.na.iytep===t&&this.la(this.O.na,Number.MIN_VALUE,Number.MAX_VALUE)},n}(fn),pn={ietupmochctefno:new ln(vn,j).l},yn=_(sn,[{i_irurefed:S,ii_irueniln:S,iart_iruremrofsn:j,itcelloc_irustats_frep_:S,i_iruepyt:S}]),Ln=new ln(fn,yn).l,Rn=function(){function t(t){void 0===t&&(t={}),this.I=t}return t.prototype.tagName=function(t){return this.I&&this.I.TagName===t},t}(),gn=function(){function t(t){this.I=t||""}return t.prototype.pattern=function(t){return Z(this.I,t)},t}(),Nn=function(){function t(t){this.Pa=t||{}}return t.prototype.contentType=function(t){return this.Pa.ContentType===t},t}(),Dn=new ln(function(t){void 0===t&&(t={}),this.uri=new fn(t.uri),this.tag=new Rn(t.tag),this.str=new gn(t.str),this.meta=new Nn(t.meta)},_(sn,[{i_irurefed:S,ii_irueniln:S,iart_iruremrofsn:j,itcelloc_irustats_frep_:S,i_iruepyt:S,imrofsnartelbatnoita:j,irttnetnocremrofsna:j,ini_gatnoitcej:j}])).l,wn=function(){function t(t){this.Va=t=t||{}}return t.prototype.U=function(t){var n=this.Va.headers;return n&&n[t]||null},t.prototype.va=function(t,n){return F(this.U(t),n)},t}(),En=i({},pn,{isetupmoctcejbobu:Dn,ibusetupmociruecruoser:Ln,ixorpetupmocesnopsernoy:new ln(wn,j).l,iunoyxorpetupmocesnopsermaertsp:new ln(wn,j).l,idleihsdaetupmoctseuqerdedocedno:new ln(vn,j).l,inoxotobetupmoctseuqerdedoced:new ln(function(){},j).l,ilcetupmoctnevetnei:new ln(function(t){void 0===t&&(t={}),this.I=t},j).l}),Sn=function(t){function n(n,i,e){return t.call(this,function(t){this.env=new n(t.env),this.req=new i(t.req),this.inp=new i(t.inp)},e,En)||this}return s(n,t),n}(ln),Yn=function(){function t(t){void 0===t&&(t={}),this.B=t}return t.prototype.applicationName=function(t){return this.B.applicationName===t},t.prototype.applicationVersion=function(t){return this.B.applicationVersion===t},t.prototype.instanceName=function(t){return t===this.B.instanceName},t.prototype.attribute=function(t,n){return"object"==typeof this.B.attributes&&this.B.attributes[t]===n},t}(),_n=Y({},[{Version:function(t){return console.error("Version cannot be changed from "+t),t},ilgo:j,itemscir:j},{ireggolgifnoc:T,itreggoltropsnar:j,icirtemgifnoc:T,itcirtemtropsnar:S,if_nohcte:H}]);_(sn,[_n]);var jn=_(an,[{ieilc_notneve_tn:H}]),Tn=_(an,[{iedoced_notseuqer_d:H}]),Hn=_(an,[{iedoced_notseuqer_d:H}]),Wn=_(j,[{ier_noesnops:H,iaertspu_noesnopser_m:H,irapyreuqecalperma:an}]),Fn=_(sn,[{uri:H}]),Zn=new Sn(Yn,vn,_(sn,[_n,{inv:jn,igolreg:j,ihsdadlei:Tn,iobxot:Hn,impp:S,ieis:S,irpyxo:Wn,irmu:j,ieikooctcelfer:j,iciruotpyr:S,iletdayrteme:S,iosivonanecivres_r:S,imarfieerte:S,ierbusecruos:Fn,iobustcejb:H,i_emonececivres:S,ipa_emonececivres_i:S,iipa_cabrecivres_:S}])),Mn=Zn.ComputeAll;Zn.pa=function(i,e){(i.req.b("chrome",46,1e3)||i.req.b("safari",10,1e3))&&e({iadb:{ierelbanenoitcerid:!0}}),(i.req.b("chrome",46,1e3)||i.req.b("firefox",38,1e3)||i.req.b("safari",10,1e3)||i.req.b("ie",10,1e3)&&document&&10>document.documentMode||i.req.b("edge",12,1e3))&&e({iadb:{idelbanenoitcete:!0}}),e({iadb:{ivetatselbaira:"MorphState",iolbetatseulavdekc:"blocked",ilbtonetatseulavdekco:"not-blocked",iycnetalelbairav:"Abdt",iircsdekcolbmunsrotcetedkniltp:4,isdekcolbnamregmunsrotcetedkniltpirc:0,imthdekcolbmunsrotcetedknill:4,ihtnemelemunsrotcetededi:4,iidekcolbmunsrotcetedgm:4,inoitceteddlohserht:3,ilufelbanenoitcetedl:!0,iyrotsihelbanenoitceriderrof:!1,irotsihmundlohserhty:1,iawirudatsilhct:[/(?:pubads[.])|(?:prebid[.]js)|(?:openx[.]net)|(?:googlesyndication[.](?:net|com))|(?:indexww[.]com)|(?:[.]adnxs[.]com)/i],itemeletdbatniopdneyr:"/g00/2_d3d3LmJvc3Rvbi5jb20%3D_/TU9SRVBIRVVTOCRodHRwOi8vY3AtaW4ubmFub3Zpc29yLmlvL2NsaWVudHByb2ZpbGVyL2FkYj9pMTBjLm1hcmsuc2NyaXB0LnR5cGU%3D_$/$/$",iceriderelbasidnoissesrofnoit:!1,isabnoitceriderelbanesrotcetedkniltpircsde:!0,ietedtratsemitnoitc:0,iofetavitcarekcolbynar:!0}}),i.req.b("safari",10,1e3)&&e({iadb:{icolbdaslru_k:"https://s0.2mdn.net/ads/richmedia/studio/pv2/60282074/20170720084443516/script.js https://cdn.doubleverify.com/dv-match4.js https://cdn.springserve.com/vd/vd0.2.82.8.js https://cdn.revcontent.com/build/css/rev2.min.css https://c.betrad.com/a/n/269/635.js https://s0.2mdn.net/6440533/1495124845208/Raise%20Your%20Hands_728x90/global.min.css".split(" ")}}),(i.req.b("firefox",38,1e3)||i.req.b("safari",10,1e3)||i.req.b("ie",10,1e3)&&document&&10>document.documentMode||i.req.b("edge",12,1e3))&&e({iadb:{ilufelbanenoitcetedl:!1,iyrotsihelbanenoitceriderrof:!1,iceriderelbasidnoissesrofnoit:!1}}),(i.req.domain(t.atob("d3d3LmxpZmV3aXJlLmNvbQ=="))||i.req.domain(t.atob("d3d3LnRoZWJhbGFuY2UuY29t"))||i.req.domain(t.atob("d3d3LnRoZXNwcnVjZS5jb20="))||i.req.domain(t.atob("d3d3LnRob3VnaHRjby5jb20="))||i.req.domain(t.atob("d3d3LnRyaXBzYXZ2eS5jb20=")))&&(e({iadb:{iyrotsihelbanenoitceriderrof:!1,isabnoitceriderelbanesrotcetedkniltpircsde:!1,ilufelbanenoitcetedl:!1,ihtnemelemunsrotcetededi:0,iidekcolbmunsrotcetedgm:0,ietedtratsemitnoitc:300}}),i.req.cb()&&(i.req.R("300")&&e({iadb:{ietedtratsemitnoitc:300}}),i.req.R("500")&&e({iadb:{ietedtratsemitnoitc:500}}),i.req.R("dcl")&&e({iadb:{ietedtratsemitnoitc:"DOMContentLoaded"}}),i.req.R("load")&&e({iadb:{ietedtratsemitnoitc:"load"}}))),i.req.a(t.atob("cG9wc2NpLmNvbQ=="))&&e({iadb:{isabnoitceriderelbanesrotcetedkniltpircsde:!1}}),!i.req.a(t.atob("Y25ldC5jb20="))||i.req.host(t.atob("d3d3LmNuZXQuY29t"))||i.req.host(t.atob("dGVzdC5jbmV0LmNvbQ=="))||i.req.host(t.atob("ZG93bmxvYWQuY25ldC5jb20="))||i.req.host(t.atob("ZG93bmxvYWQuc3RhZ2UuY25ldC5jb20="))||e({iadb:{idelbanenoitcete:!1}}),i.req.b("firefox")&&i.req.host(t.atob("dGVzdC5jbmV0LmNvbQ=="))&&e({iadb:{ierelbanenoitcerid:!0}}),i.req.host(t.atob("d3d3LmNuZXQuY29t"))&&e({iadb:{inietatseulavti:"detecting"}}),i.req.a(t.atob("aG9tZXMuY29t"))&&i.req.b("firefox",54,1e3)&&e({iadb:{ierelbanenoitcerid:!0}}),i.req.a(t.atob("ZWRtdW5kcy5jb20="))&&i.req.b("firefox",54,1e3)&&e({iadb:{ierelbanenoitcerid:!0}}),i.req.a(t.atob("dHJlbmQtY2hhc2VyLmNvbQ=="))&&i.req.b("firefox",54,1e3)&&e({iadb:{ierelbanenoitcerid:!0}}),i.req.host(t.atob("d3d3LmxvbGtpbmcubmV0"))&&e({iadb:{ilufelbanenoitcetedl:!0,iyrotsihelbanenoitceriderrof:!0,iawirudatsilhct:[/(?:mb[.]zam)/i]}}),i.req.host(t.atob("d3d3LmxvbGtpbmcubmV0"))&&i.req.cookie("i10cfd","1")&&e({iadb:{iyrotsihelbanenoitceriderrof:!0,iawirudatsilhct:[/(?:mb[.]zam)/i]}}),(i.req.host(t.atob("d3d3Lm1zbi5jb20="))||i.req.host(t.atob("aW50MS5tc24uY29t")))&&e({iadb:{ivetatselbaira:"Abd",iolbetatseulavdekc:1,ilbtonetatseulavdekco:0,isabnoitceriderelbanesrotcetedkniltpircsde:!1,itemeletdbatniopdneyr:"//tu9srvbirvvtocrjcc1pbi5uyw5vdmlzb3iuaw80.g00.msn.com/g00/2_d3d3LmJvc3Rvbi5jb20%3D_/TU9SRVBIRVVTOCRodHRwOi8vY3AtaW4ubmFub3Zpc29yLmlvL2NsaWVudHByb2ZpbGVyL2FkYj9pMTBjLm1hcmsuc2NyaXB0LnR5cGU%3D_$/$/$"}}),(i.req.host(t.atob("d3d3Lm1zbi5jb20="))||i.req.host(t.atob("aW50MS5tc24uY29t")))&&(i.req.b("firefox")||i.req.b("edge"))&&e({iadb:{ierelbanenoitcerid:!0}}),i.req.host(t.atob("d3d3Lm1zbi5jb20="))&&(i.req.b("chrome",46,1e3)||i.req.b("safari",10,1e3))&&e({iadb:{ilufelbanenoitcetedl:!0,ierelbanenoitcerid:!0,iyrotsihelbanenoitceriderrof:!0,iceriderelbasidnoissesrofnoit:!1,iawirudatsilhct:[/(?:cdn[.]3lift[.]com)|(?:ib[.]3lift[.]com\/ttj)|(?:widgets[.]outbrain[.]com\/external\/publishers\/msn\/MSNOBCore[.]min[.]js)|(?:cdn[.]taboola[.]com\/libtrc\/msn-home-network\/loader[.]js)|(?:cdn[.]taboola[.]com\/libtrc\/msn-section-network\/loader[.]js)|(?:h6[.]msn[.]com\/nativeads\/ms-nativeads-msn[.]min[.]js)|(?:at[.]atwola[.]com)|(?:pixel[.]advertising[.]com)/i]}}),(i.req.host(t.atob("d3d3Lm1zbi5jb20="))||i.req.host(t.atob("aW50MS5tc24uY29t")))&&t&&1===t["ad-instart2"]&&e({iadb:{ierelbanenoitcerid:!1}}),i.req.host(t.atob("bnYuY29t"))&&e({iadb:{iircsdekcolbmunsrotcetedkniltp:4,isdekcolbnamregmunsrotcetedkniltpirc:4,imthdekcolbmunsrotcetedknill:0,ihtnemelemunsrotcetededi:4,iidekcolbmunsrotcetedgm:4,inoitceteddlohserht:16,ierelbanenoitcerid:!1,ilufelbanenoitcetedl:!1,iyrotsihelbanenoitceriderrof:!1}}),(i.req.a(t.atob("Y2FsZ2FyeWhlcmFsZC5jb20="))||i.req.a(t.atob("Y2FsZ2FyeXN1bi5jb20="))||i.req.a(t.atob("Y2FuYWRhLmNvbQ=="))||i.req.a(t.atob("Y2Fub2UuY29t"))||i.req.a(t.atob("ZWRtb250b25qb3VybmFsLmNvbQ=="))||i.req.a(t.atob("ZWRtb250b25zdW4uY29t"))||i.req.a(t.atob("ZmFjZW9mZi5jb20="))||i.req.a(t.atob("ZmluYW5jaWFscG9zdC5jb20="))||i.req.a(t.atob("aG9ja2V5aW5zaWRlb3V0LmNvbQ=="))||i.req.a(t.atob("bGVhZGVycG9zdC5jb20="))||i.req.a(t.atob("bGZwcmVzcy5jb20="))||i.req.a(t.atob("bW9udHJlYWxnYXpldHRlLmNvbQ=="))||i.req.a(t.atob("bmF0aW9uYWxwb3N0LmNvbQ=="))||i.req.a(t.atob("b3R0YXdhY2l0aXplbi5jb20="))||i.req.a(t.atob("b3R0YXdhc3VuLmNvbQ=="))||i.req.a(t.atob("dGhlZ2lmdGd1aWRlLmNh"))||i.req.a(t.atob("dGhlcHJvdmluY2UuY29t"))||i.req.a(t.atob("dGhlc3RhcnBob2VuaXguY29t"))||i.req.a(t.atob("dG9yb250b3N1bi5jb20="))||i.req.a(t.atob("dmFuY291dmVyc3VuLmNvbQ=="))||i.req.a(t.atob("d2luZHNvcnN0YXIuY29t"))||i.req.a(t.atob("d2lubmlwZWdzdW4uY29t")))&&e({iadb:{ilufelbanenoitcetedl:!1,iyrotsihelbanenoitceriderrof:!1}}),i.req.host(t.atob("d3d3LnJhbmtlci5jb20="))&&(i.req.wa("mac")||i.req.wa("windows"))&&e({iadb:{isabnoitceriderelbanesrotcetedkniltpircsde:!1,ierelbanenoitcerid:!0,ilufelbanenoitcetedl:!0,iyrotsihelbanenoitceriderrof:!0}}),i.req.domain(t.atob("d3d3LnN1cGVyY2hldnkuY29t"))&&e({iadb:{iyrotsihelbanenoitceriderrof:!1,isabnoitceriderelbanesrotcetedkniltpircsde:!1,ilufelbanenoitcetedl:!1,ihtnemelemunsrotcetededi:0,iidekcolbmunsrotcetedgm:0,ietedtratsemitnoitc:300}}),(i.req.host(t.atob("dHZ0cm9wZXMub3Jn"))||i.req.host(t.atob("d3d3LnR2dHJvcGVzLm9yZw==")))&&e({iadb:{iofetavitcarekcolbynar:!1,ifetavitcakcolbdaro:!0,irofetavitcasulpkcolbda:!0}}),i.req.a(t.atob("d2FzaGluZ3RvbnBvc3QuY29t"))&&e({iadb:{ilufelbanenoitcetedl:!1,ietedtratsemitnoitc:"DOMContentLoaded",iotelbairavlabolgskcabllacretsiger:"__il_wapo",ierelbanenoitcerid:!1}}),n(i,e),i.req.host(t.atob("d3d3Lndvd2hlYWQuY29t"))&&e({iadb:{ilufelbanenoitcetedl:!0,iyrotsihelbanenoitceriderrof:!0,iawirudatsilhct:[/(?:mb[.]zam)/i]}}),i.req.host(t.atob("d3d3Lndvd2hlYWQuY29t"))&&i.req.cookie("i10cfd","1")&&e({iadb:{iyrotsihelbanenoitceriderrof:!0,iawirudatsilhct:[/(?:mb[.]zam)/i]}}),i.req.b("chrome",46,1e3)&&e({iadb:{iidekcolbmunsrotcetedgm:0}}),i.req.ab(cn)&&e({iadb:{idelbanenoitcete:!1}}),i.req.cookie("i10c.expt.history","true")&&e({iadb:{iyrotsihelbanenoitceriderrof:!0}})};var kn,Vn=null,On=kn={url:t.location.href,headers:{host:t.location.host,cookie:document.cookie}};Vn=Mn({env:{},inp:On,req:kn},{});var xn,An=function(){function t(t,n){this.J=t,this.Za=n}return t.prototype.iulr=function(){return this.Za.url},t.prototype.sa=function(){return 1223===this.J.status?204:this.J.status},t.prototype.Headers=function(){return this.J.getAllResponseHeaders()},t.prototype.iaehred=function(t){return this.J.getResponseHeader(t)},t.prototype.Error=function(){return 100>this.sa()||599<this.sa()?"Network request failed":null},t.prototype.Text=function(){return this.J.responseText},t}(),Cn={Da:XMLHttpRequest,Sa:XMLHttpRequest.prototype.open,Wa:XMLHttpRequest.prototype.send},Bn=(xn={},xn[Zt]=rt,xn[Tt]=!1,xn);t.addEventListener("message",function(t){"string"!=typeof(t=t.data)||0!==t.lastIndexOf(Mt,Mt.length+1)&&0!==t.lastIndexOf(kt,kt.length+1)||(Bn[Tt]=!0,Bn[Ht]=t)});var Gn;!function(t){t[t.PENDING=0]="PENDING",t[t.FULLFILLED=1]="FULLFILLED",t[t.REJECTED=2]="REJECTED"}(Gn||(Gn={}));var Xn=function(){function t(t){if(!t)throw"no executor provided";if("object"!=typeof this)throw"use new";if("function"!=typeof t)throw"not a function";this.state=Gn.oa,this.Ja(t)}return t.prototype.Ja=function(t){var n=this;try{t(function(t){n.Ua(t)},function(t){n.ra(t)})}catch(t){n.ra(t)}},t.prototype.Ua=function(t){this.xa&&this.xa.call(this,t),this.state=Gn.ha},t.prototype.ra=function(t){this.fa&&this.fa.call(this,t),this.state=Gn.qa},t.prototype.then=function(n,i){return this.state===Gn.oa?(this.fa=i,this.xa=n):this.state===Gn.ha?n.call(this,void 0):this.state===Gn.qa&&i.call(this,void 0),new t(function(){})},t.prototype.catch=function(n){return this.state===Gn.oa?this.fa=n:this.state!==Gn.ha&&this.state===Gn.qa&&n.call(this,void 0),new t(function(){})},t.prototype.all=function(n){var i=Array.prototype.slice.call(n);return new t(function(t,n){function e(o,u){try{if(u&&("object"==typeof u||"function"==typeof u)){var c=u.then;if("function"==typeof c)return void c.call(u,function(t){e(o,t)},n)}i[o]=u,0==--r&&t(i)}catch(t){n(t)}}if(0===i.length)return t([]);for(var r=i.length,o=0;o<i.length;o++)e(o,i[o])})},t}(),zn=(new nn).idtesniamo(t.location.hostname).ioortegniamodt(),Qn={},In="aHR0cHM6Ly9wYWdlYWQyLmdvb2dsZXN5bmRpY2F0aW9uLmNvbS9mYXZpY29uLmljbw== aHR0cHM6Ly9zMC4ybWRuLm5ldC8xNjM1OTA5LzF4MWltYWdlLmpwZw== aHR0cHM6Ly9jb25uZWN0LmZhY2Vib29rLm5ldC9mYXZpY29uLmljbw== aHR0cHM6Ly9hZHMudHdpdHRlci5jb20vZmF2aWNvbi5pY28= aHR0cHM6Ly93d3cuZ29vZ2xlLWFuYWx5dGljcy5jb20vX191dG0uZ2lm aHR0cHM6Ly90cGMuZ29vZ2xlc3luZGljYXRpb24uY29tL2Zhdmljb24uaWNv aHR0cHM6Ly9zZWN1cmUuZm9vdHByaW50Lm5ldC95aWVsZG1hbmFnZXIvYXBleC9tZWRpYXN0b3JlL2FkY2hvaWNlXzEucG5n".split(" "),Pn="aHR0cHM6Ly9jZG4uZmxhc2h0YWxraW5nLmNvbS83MTE5NS8xODkwMTQ2L2pzL3Rpbnlyb29tLmpz aHR0cHM6Ly9jZG4ubWVkaWF2b2ljZS5jb20vbmF0aXZlYWRzL3NjcmlwdC9jYWZlbW9tL3BvbGFyX3RoZXN0aXJfcGlwaW5nLmpz aHR0cHM6Ly9jLmJldHJhZC5jb20vYS9uLzI3OS83ODE3NS5qcw== aHR0cHM6Ly9zY3JpcHRzLmhvc3QuYmFubmVyZmxvdy5jb20vMS4wLjAvd2lkZ2V0Lm1pbi5qcw== aHR0cHM6Ly9zdGF0aWMuZG91YmxlY2xpY2submV0L2luc3RyZWFtL2FkX3N0YXR1cy5qcw== aHR0cHM6Ly9zMC4ybWRuLm5ldC82OTU1NTEyLzE0OTkzNDQxOTMxMjMvd2ViLzMwMHgyNTBfZWRnZS5qcw==".split(" "),Un="aHR0cHM6Ly9ydGF4LmNyaXRlby5jb20vZGVsaXZlcnkvcnRhL3J0YS5qcw== aHR0cHM6Ly9pYi5hZG54cy5jb20vanB0 aHR0cHM6Ly90bHguM2xpZnQuY29tL2hlYWRlci9hdWN0aW9u aHR0cHM6Ly9hZC55aWVsZGxhYi5uZXQveXAvNTE5NDc4 aHR0cHM6Ly9wYWdlYWQyLmdvb2dsZXN5bmRpY2F0aW9uLmNvbS9wYWdlYWQvanMvZ29vZ2xlX3RvcF9leHAuanM= aHR0cHM6Ly9zZWN1cmUuYWRueHMuY29tL3R0ag==".split(" "),Jn=new U,$n=o()&&r()&&!d();Jn.i($n);var qn=void 0,Kn=new U,ti=o()&&(void 0!==qn?qn:qn=!!t[q]);Kn.i(ti);var ni=t.document.currentScript||t.document.scripts[document.scripts.length-1],ii=t.Promise||Xn,ei="&ad_box_ &ad_channel= &ad_classid= &ad_height= &ad_keyword= &ad_network_ &ad_number= &ad_type= &ad_type_ &ad_url= &ad_zones= &adbannerid=".split(" "),ri=[["iframe","google_ads_frame"],["iframe","google_ads_iframe"],["div","ADV-SLOT-"],["div","YFBMSN"],["div","google_dfp_"],["div","MarketGid"]],oi=function(){function n(t){this.F={},this.Config=t,this.o={D:0,Z:0,Y:0},this.Qa=this.Ma(),this.ba=[]}return n.prototype.f=function(t){return this.F[t]},n.prototype.La=function(){var t=[];t.push(O(Rt,!0,JSON.stringify(this.o)));var n,i=A();for(n in i)t.push(O(n,i[n].w,"STORED:"+JSON.stringify(i[n].A)));i=this.F,Qn=A();for(var e in i)Qn&&Qn[e]?Qn[e].w===i[e]?Qn[e].A+=1:(Qn[e].w=i[e],Qn[e].A=1):Qn[e]={w:i[e],A:1};for(e in Qn)i[e]||delete Qn[e];if(e=JSON.stringify(Qn),C(Vt))try{sessionStorage._306_6624720851103629=e}catch(t){}return x(document.cookie,Ot)||(document.cookie=Ot+"=1; domain="+zn+"; path=/"),t},n.prototype.ja=function(){for(var n=[],i=new Date,e=1;6>=e;e++){var r=e+":"+i.getMonth()+":"+e+":"+i.getDate()+":"+e+":"+i.getHours()%2+":"+e,o=xt+"="+e+"&ad_channel=1";r="//hxyzhas.g00."+new nn(t.location.href).ioortegniamodt()+_t+"/"+encodeURIComponent(t.btoa(r))+"/ad?"+o,n.push(r)}return n},n.prototype.Ma=function(){for(var t=[],n=this.Config.iadb.icolbdaslru_k?this.Config.iadb.icolbdaslru_k.slice():this.Config.iadb.isabnoitceriderelbanesrotcetedkniltpircsde?this.ja():Pn.slice(),i=Un.slice(),e=0;e<this.Config.iadb.iircsdekcolbmunsrotcetedkniltp;e++)t.push(G("track",n,!1,o(),!1));for(e=0;e<this.Config.iadb.isdekcolbnamregmunsrotcetedkniltpirc;e++)t.push(G("track",i,void 0,void 0,void 0));for(e=0;e<this.Config.iadb.ihtnemelemunsrotcetededi;e++)t.push(X());for(e=0;e<this.Config.iadb.iidekcolbmunsrotcetedgm;e++)t.push(B());return this.o.D=t.length,t},n.prototype.Na=function(){var t=[],n=this.Config.iadb.isabnoitceriderelbanesrotcetedkniltpircsde?this.ja():Pn.slice();0<this.Config.iadb.isdekcolbnamregmunsrotcetedkniltpirc&&(n=n.concat(Un));for(var i=0;i<n.length;i++)t.push(G("track",n,void 0,void 0,void 0));return this.o.D+=t.length,t},n.prototype.Ea=function(t){this.Ha(t),this.Fa(t),this.Ga(t)},n.prototype.Ha=function(t){function n(n,e){i.F[n]=e,t(O(n,e))}var i=this;Ct.u(function(t){return n(it,t)}),Kn.u(function(t){return n(ot,t)}),Jn.u(function(t){return n(nt,t)})},n.prototype.Fa=function(t){for(var n=this,i=0;i<this.o.D;i++)this.Qa[i].then(function(i){return n.ga(i,t)})},n.prototype.Ga=function(n){var i=this;this.Config.iadb.ilufelbanenoitcetedl&&(this.ba=this.Na(),t.addEventListener("load",function(){for(var t=0;t<i.ba.length;t++)i.ba[t].then(function(t){return i.ga(t,n)})}))},n.prototype.ga=function(t,n){if(this.o.Z++,t[Tt]){var i=++this.o.Y;!this.F[Lt]&&i>=this.Config.iadb.inoitceteddlohserht&&(this.F[Lt]=!0)}n(t)},n}();[In,Pn,Un].forEach(function(n){return n.forEach(function(n,i,e){return e[i]=t.atob(n)})}),!t[$]&&R($,{}),!t[J]&&R(J,{});var ui=function(){function n(t){var n;this.h=(n={},n[Tt]=!1,n[Ft]=0,n[Wt]=[],n),this.Ia=Date.now(),this.$=this.da=this.K=!1,this.Config=t,this.c=new oi(t),t.iadb.iceriderelbasidnoissesrofnoit&&!x(document.cookie,"_306_6624720851103629")&&(document.cookie="_306_6624720851103629=1; domain="+zn+"; path=/"),this.T=this.Config.iadb.iyrotsihelbanenoitceriderrof&&this.za()}return n.prototype.V=function(){var t=this.Config.iadb.ierelbanenoitcerid;return x(document.cookie,"_306_6624720851103629")&&(t=!1),t},n.prototype.Ta=function(){var t,n=this.$=!0;try{g(this.Config.iadb.ivetatselbaira,this.K?this.Config.iadb.iolbetatseulavdekc:this.Config.iadb.ilbtonetatseulavdekco),g(this.Config.iadb.iycnetalelbairav,this.h[Ft]);var i=D(this.h,this.Config.iadb.iotelbairavlabolgskcabllacretsiger);this.K&&!this.c.f(ot)&&!this.c.f(nt)&&i&&this.V()&&(this.N(!1),this.aa()),n=!1}catch(i){this.h[Wt].push((t={},t[Zt]=et,t[Tt]=!0,t)),this.N(!0),this.V()&&this.aa(),n=!1}finally{n&&this.V()&&this.aa()}},n.prototype.N=function(n){if(this.da){this.da=!0,this.Aa(n),n=this.Config.iadb.itemeletdbatniopdneyr;var i=this.h;if(n)try{var e=[];e=i[Wt]&&(i[Tt]?i[Wt]:i[Wt].filter(V)),Bn[Tt]&&e.push(Bn);var r={adbType:[i[Tt]?Nt:Dt],otherData:JSON.stringify(e),clientIpAddr:"",userAgent:navigator.userAgent,pageUrl:t.location.href,detectionTime:i[Ft]},o={data_stream_type:"adb",json_data:JSON.stringify(r),generated_timestamp_msec:Date.now()};t.navigator.sendBeacon?t.navigator.sendBeacon(n,JSON.stringify(o)):k({method:"POST",url:n,body:o,async:!0,onload:void 0,onerror:void 0,onchange:void 0,S:[{ianem:"Content-type",Ya:"text/plain"}]})}catch(t){}}},n.prototype.Aa=function(t){var n,i;this.Config.iadb.iyrotsihelbanenoitceriderrof&&this.h[Wt].push((n={},n[Zt]=pt,n[Tt]=!0,n)),this.Config.iadb.ilufelbanenoitcetedl&&(this.h[Wt].push((i={},i[Zt]=vt,i[Tt]=!0,i)),t&&(this.h[Wt]=this.h[Wt].concat(this.c.La())))},n.prototype.za=function(){var t,n=A(),i=this.Config.iadb.irotsihmundlohserhty;return n[ht]&&n[ht].w&&n[ht].A>=i&&(t=ht),n[Lt]&&n[Lt].w&&n[Lt].A>=i&&(t=t?t+":"+Lt:Lt),n[at]&&n[at].w&&(t=t?t+":"+at:at),t},n.prototype.Ba=function(){var t=!1;if(this.$){var n=this.c.o;n.Z!==n.D||this.Config.iadb.ilufelbanenoitcetedl&&(void 0===this.c.f(bt)||void 0===this.c.f(dt)||void 0===this.c.f(lt)||void 0===this.c.f(ft)||void 0===this.c.f(ht)||void 0===this.c.f(at)||void 0===this.c.f(yt))||(t=!0)}return t},n.prototype.Oa=function(){var t,n=this.Config.iadb.iircsdekcolbmunsrotcetedkniltp+this.Config.iadb.ihtnemelemunsrotcetededi+this.Config.iadb.iidekcolbmunsrotcetedgm+this.Config.iadb.isdekcolbnamregmunsrotcetedkniltpirc,i=this.c.o,e=this.Config.iadb.inoitceteddlohserht,r=void 0!==this.c.f(it)&&void 0!==this.c.f(ot)&&void 0!==this.c.f(nt);return!0===this.c.f(it)||!0===this.c.f(ot)||!0===this.c.f(nt)||i.Y>=e||this.T&&r?(this.T&&this.h[Wt].push((t={},t[Zt]=gt,t[Tt]=!0,t[Ht]=this.T,t)),Nt):n===i.Z&&i.Y<e&&!1===this.c.f(it)?Dt:wt},n.prototype.ya=function(){var n=this;t.addEventListener("unload",function(){return n.N(!0)}),this.c.Ea(function(t){n.h[Wt].push(t),n.$?!n.da&&n.Ba()&&n.N(!0):(t=n.Oa())!==wt&&(n.K=t===Nt,n.h[Tt]=n.K,n.h[Ft]=Date.now()-n.Ia,n.Ta())})},n.prototype.aa=function(){p()},n}();if(t===top&&!new RegExp(jt,"i").test(t.location.pathname)&&Vn.iadb.idelbanenoitcete){var ci=function(){new ui(Vn).ya()},si=Vn.iadb.ietedtratsemitnoitc;si?"string"==typeof si?t.addEventListener(si,ci):t.setTimeout(ci,si):ci()}}();try{t.INSTART.Init(null)}catch(t){}}}(window);
    	}
    </script>
    <div class="background-image-gradient"></div>
    <!-- Wikia Beacon Tracking -->
    <script>
    	require(['wikia.trackingOptIn'], function (trackingOptIn) {
    		trackingOptIn.pushToUserConsentQueue(function (optIn) {
    			function getCookieValue(cookieName) {
    				var cookieSplit = ('; ' + document.cookie).split('; ' + cookieName + '=');
    
    				return cookieSplit.length === 2 ? cookieSplit.pop().split(';').shift() : null;
    			}
    
    			var script = document.createElement('script'),
    				utma = getCookieValue('__utma'),
    				utmb = getCookieValue('__utmb'),
    				trackUrl;
    
    			trackUrl = "https://beacon.wikia-services.com/__track/view?cb=7800027800012&c=43339&lc=en&lid=75&x=lyricwiki&y=c1&a=275575&s=oasis&&n=0" + ((typeof document.referrer != "undefined") ? "&r=" + encodeURIComponent(document.referrer) : "") +
    					"&rand=" + (new Date).valueOf() + (window.beacon_id ? "&beacon=" + window.beacon_id : "") +
    					(utma && utma[1] ? "&utma=" + utma[1] : "") + (utmb && utmb[1] ? "&utmb=" + utmb[1] : "") +
    					'&session_id=' + window.sessionId + '&pv_unique_id=' + window.pvUID + '&pv_number=' + window.pvNumber +
    					'&pv_number_global=' + window.pvNumberGlobal;
    
    			if (optIn) {
    				
    				trackUrl += '&u=' + '0';
    			} else {
    				trackUrl += '&u=-1';
    			}
    
    			script.src = trackUrl;
    			document.head.appendChild(script);
    		});
    	});
    </script>
    <script>
    	require(['wikia.trackingOptIn'], function (trackingOptIn) {
    		trackingOptIn.init();
    	});
    </script>
    <!-- Begin comScore Tag -->
    <script type="text/javascript">
    require(["wikia.trackingOptIn"], function (trackingOptIn) {
    	function loadComscoreScript() {
    		window._comscore = window._comscore || [];
    		window._comscore.push({ c1: "2", c2: "6177433",
    			options: {
    				url_append: "comscorekw=wikiacsid_lyrically"
    			}
    		});
    	
    		var s = document.createElement("script");
    		s.async = true;
    		s.src = (document.location.protocol == "https:" ? "https://sb" : "http://b") + ".scorecardresearch.com/beacon.js";
    		document.head.appendChild(s);
    	}
    
    	trackingOptIn.pushToUserConsentQueue(function (optIn) {
    		if (optIn) {
    			loadComscoreScript();
    		}
    	});
    });
    </script>
    <!-- End comScore Tag -->
    <!-- Start for QuantServe, page_view -->
    <script type="text/javascript">
    window._qevents = window._qevents || [];
    require(["wikia.trackingOptIn"], function (trackingOptIn) {
    	function loadQuantServeScript() {
    		var elem = document.createElement('script');
    		
    		elem.src = (document.location.protocol == "https:" ? "https://secure" : "http://edge") + ".quantserve.com/quant.js";
    		elem.async = true;
    		elem.type = "text/javascript";
    		
    		document.head.appendChild(elem);
    	}
    
    	trackingOptIn.pushToUserConsentQueue(function (optIn) {
    		if (optIn) {
    			loadQuantServeScript();
    		}
    	});
    });
    </script>
    <script type="text/javascript">
    var quantcastLabels = "";
    if (window.wgWikiVertical) {
    	quantcastLabels += wgWikiVertical;
    	if (window.wgDartCustomKeyValues) {
    		var keyValues = wgDartCustomKeyValues.split(';');
    		for (var i=0; i<keyValues.length; i++) {
    			var keyValue = keyValues[i].split('=');
    			if (keyValue.length >= 2) {
    				quantcastLabels += ',' + wgWikiVertical + '.' + keyValue[1];
    			}
    		}
    	}
    }
    _qevents.push( { qacct:"p-8bG6eLqkH6Avk", labels:quantcastLabels } );
    </script>
    <!-- Start for BillTheLizard, page_view -->
    <script>
    	require([
    		require.optional('ext.wikia.adEngine.ml.billTheLizard')
    	], function (billTheLizard) {
    		if (billTheLizard) {
    			billTheLizard.call();
    		}
    	});
    </script>
    <!-- Start for MoatYi, page_view -->
    <script>
    	require([
    		require.optional('ext.wikia.adEngine.tracking.moatYi')
    	], function (moatYi) {
    		if (moatYi) {
    			moatYi.call();
    		}
    	});
    </script>
    <!-- Start for A9, page_view -->
    <script>
    	require([
    		'ext.wikia.adEngine.adContext',
    		'ext.wikia.adEngine.lookup.a9'
    	], function (adContext, a9) {
    		if (adContext.get('bidders.a9') && !adContext.get('bidders.prebidAE3')) {
    			a9.call();
    		}
    	});
    </script>
    <!-- Start for Prebid, page_view -->
    <script>
    	require([
    		'ext.wikia.adEngine.adContext',
    		'ext.wikia.adEngine.lookup.prebid'
    	], function (adContext, prebid) {
    		if (adContext.get('bidders.prebid') && !adContext.get('bidders.prebidAE3')) {
    			prebid.call();
    		}
    	});
    </script>
    <!-- Begin Krux Tag -->
    <script type="text/javascript">
    	require(['wikia.krux'], function (krux) {
    		krux.load('JU3_GW1b');
    	});
    </script>
    <!-- End Krux Tag -->
    <!-- Begin NetzAthleten (Netletix) Tag -->
    <script type="text/javascript">
    require([
    	'ext.wikia.adEngine.adContext',
    	'wikia.geo',
    	'wikia.instantGlobals',
    	'wikia.trackingOptIn'
    ], function(adContext, geo, instantGlobals, trackingOptIn) {
    	var url = "//s.adadapter.netzathleten-media.de/API-1.0/NA-828433-1/naMediaAd.js";
    
    	function loadNetzAthletenScript() {
    		var scriptElementNetzAthleten = document.createElement("script");
    		scriptElementNetzAthleten.type = "text/javascript";
    		scriptElementNetzAthleten.src = url;
    
    		scriptElementNetzAthleten.addEventListener("load", function() {
    			window.naMediaAd.setValue("homesite", window.wgIsMainpage);
    		});
    		document.head.appendChild(scriptElementNetzAthleten);
    	}
    
    	function loadNA(optIn) {
    		if (optIn && adContext.get('opts.netzathleten')) {
    			// user opt-ins for NetzAthleten
    			loadNetzAthletenScript();
    		}
    	}
    
    	if (trackingOptIn) {
    		trackingOptIn.pushToUserConsentQueue(loadNA);
    	} else {
    		loadNA(true)
    	}
    });
    </script>
    <!-- End NetzAthleten (Netletix) Tag -->
    <div class="WikiaSiteWrapper">
    <h2>FANDOM</h2>
    <div class="wikia-ad noprint" id="ad-skin"></div>
    <div class="wds-global-navigation-wrapper">
    <div class="wds-global-navigation wds-search-is-always-visible" id="globalNavigation">
    <div class="wds-global-navigation__content-container">
    <div class="wds-global-navigation__content-bar-left">
    <a class="wds-global-navigation__logo" data-tracking-label="logo" href="//fandom.wikia.com/">
    <svg class="wds-global-navigation__logo-image" id="wds-company-logo-fandom-white" viewbox="0 0 164 35" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd"><path d="M32.003 16.524c0 .288-.115.564-.32.768L18.3 30.712c-.226.224-.454.324-.738.324-.292 0-.55-.11-.77-.325l-.943-.886a.41.41 0 0 1-.01-.59l15.45-15.46c.262-.263.716-.078.716.29v2.46zm-17.167 10.12l-.766.685a.642.642 0 0 1-.872-.02L3.01 17.362c-.257-.25-.4-.593-.4-.95v-1.858c0-.67.816-1.007 1.298-.536l10.814 10.56c.188.187.505.57.505 1.033 0 .296-.068.715-.39 1.035zM5.73 7.395L9.236 3.93a.421.421 0 0 1 .592 0l11.736 11.603a3.158 3.158 0 0 1 0 4.5l-3.503 3.462a.423.423 0 0 1-.59 0L5.732 11.89a3.132 3.132 0 0 1-.937-2.25c0-.85.332-1.65.935-2.246zm13.89 1.982l3.662-3.62a3.232 3.232 0 0 1 2.737-.897c.722.098 1.378.47 1.893.978l3.708 3.667a.41.41 0 0 1 0 .585l-5.64 5.576a.419.419 0 0 1-.59 0l-5.77-5.704a.411.411 0 0 1 0-.585zm14.56-.687L26.014.475a.869.869 0 0 0-1.228-.002L18.307 6.94c-.5.5-1.316.5-1.82.004l-6.48-6.4A.87.87 0 0 0 8.793.542L.447 8.67C.16 8.95 0 9.33 0 9.727v7.7c0 .392.158.77.44 1.048l16.263 16.072a.87.87 0 0 0 1.22 0l16.25-16.073c.28-.278.438-.655.438-1.048V9.73c0-.39-.153-.763-.43-1.04z" fill="#00D6D6"></path><path d="M62.852 20.51l2.58-6.716a.468.468 0 0 1 .87 0l2.58 6.717h-6.03zm5.856-12.428c-.184-.48-.65-.8-1.17-.8h-3.342c-.52 0-.986.32-1.17.8l-7.083 18.5c-.21.552.2 1.14.796 1.14h2.753c.353 0 .67-.215.796-.542l.738-1.922a.849.849 0 0 1 .795-.542h8.088a.85.85 0 0 1 .796.542l.74 1.922c.125.327.44.543.795.543h2.754a.843.843 0 0 0 .796-1.14l-7.082-18.5zm93.504-.8h-2.715a1.86 1.86 0 0 0-1.677 1.047l-5.393 11.162-5.393-11.163a1.858 1.858 0 0 0-1.677-1.047h-2.715a.889.889 0 0 0-.893.883V26.84c0 .487.4.883.892.883h2.608a.889.889 0 0 0 .893-.883v-9.686l4.945 10.072c.15.304.46.497.803.497h1.073a.893.893 0 0 0 .803-.497l4.945-10.072v9.686c0 .487.4.883.894.883h2.608a.889.889 0 0 0 .893-.883V8.166c0-.487-.4-.883-.893-.883zm-106.972 8.8h-8.63V11.49h10.918a.88.88 0 0 0 .83-.578l.888-2.464a.872.872 0 0 0-.83-1.163h-15.18c-.486 0-.88.39-.88.87v18.7c0 .48.394.87.88.87h2.492c.486 0 .88-.39.88-.87V20.29h7.743a.88.88 0 0 0 .83-.578l.89-2.464a.872.872 0 0 0-.83-1.163zm51.76 7.61h-3.615V11.315H107c3.828 0 6.41 2.517 6.41 6.188 0 3.672-2.582 6.19-6.41 6.19zm-.124-16.41h-7.128c-.486 0-.88.39-.88.872v18.698c0 .48.394.87.88.87h7.128c6.453 0 10.912-4.44 10.912-10.16v-.117c0-5.72-4.46-10.162-10.912-10.162zm-11.947.03h-2.642a.87.87 0 0 0-.876.866v12.36l-8.755-12.72a1.242 1.242 0 0 0-1.023-.535H78.32a.873.873 0 0 0-.876.867v18.706c0 .48.393.867.877.867h2.64a.872.872 0 0 0 .878-.867V14.71l8.608 12.478c.23.334.613.535 1.022.535h3.46a.872.872 0 0 0 .877-.867V8.178a.87.87 0 0 0-.876-.867zm40.71 10.3c0 3.323-2.712 6.016-6.056 6.016-3.345 0-6.056-2.693-6.056-6.015v-.22c0-3.322 2.71-6.015 6.056-6.015 3.344 0 6.055 2.693 6.055 6.015v.22zm-6.056-10.44c-5.694 0-10.31 4.576-10.31 10.22v.22c0 5.646 4.616 10.22 10.31 10.22 5.693 0 10.308-4.574 10.308-10.22v-.22c0-5.644-4.615-10.22-10.308-10.22z" fill="#FFF"></path></g></svg> </a>
    <nav class="wds-global-navigation__links">
    <a class="wds-global-navigation__link" data-tracking-label="link.games" href="//fandom.wikia.com/topics/games">
    	Games</a>
    <a class="wds-global-navigation__link" data-tracking-label="link.movies" href="//fandom.wikia.com/topics/movies">
    	Movies</a>
    <a class="wds-global-navigation__link" data-tracking-label="link.tv" href="//fandom.wikia.com/topics/tv">
    	TV</a>
    <a class="wds-global-navigation__link" data-tracking-label="link.video" href="//fandom.wikia.com/video">
    	Video</a>
    <div class="wds-dropdown wds-global-navigation__link-group wds-has-dark-shadow">
    <div class="wds-dropdown__toggle wds-global-navigation__dropdown-toggle wds-global-navigation__link">
    <span>Wikis</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown__toggle-chevron" id="wds-icons-dropdown-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6 9l4-5H2" fill-rule="evenodd"></path></svg> </div>
    <div class="wds-dropdown__content wds-global-navigation__dropdown-content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking-label="link.explore" href="//fandom.wikia.com/explore">
    	Explore Wikis</a>
    </li>
    <li>
    <a data-tracking-label="link.community-central" href="//community.wikia.com/wiki/Community_Central">
    	Community Central</a>
    </li>
    <li>
    <a class="wds-button wds-is-secondary wds-global-navigation__link-button" data-tracking-label="link.start-a-wiki" href="//community.wikia.com/wiki/Special:CreateNewWiki">
    	Start a Wiki</a>
    </li>
    </ul>
    </div>
    </div>
    </nav>
    </div>
    <div class="wds-global-navigation__content-bar-right">
    <div class="wds-global-navigation__dropdown-controls">
    <form action="//lyrics.wikia.com/wiki/Special:Search" class="wds-global-navigation__search-container wds-search-is-focused">
    <div class="wds-dropdown wds-global-navigation__search wds-no-chevron wds-has-dark-shadow">
    <div class="wds-global-navigation__search-toggle">
    <svg class="wds-icon wds-icon-small wds-global-navigation__search-toggle-icon" id="wds-icons-magnifying-glass-small" viewbox="0 0 18 18" xmlns="http://www.w3.org/2000/svg"><g fill-rule="evenodd"><path d="M16.984 16.025l-4.03-4.043a.713.713 0 0 0-1.011 0 .72.72 0 0 0 0 1.015l4.03 4.043c.279.28.732.28 1.011 0a.72.72 0 0 0 0-1.015z"></path><path d="M2.178 7.924c0-3.17 2.56-5.74 5.72-5.74 3.16 0 5.72 2.57 5.72 5.74 0 3.17-2.56 5.739-5.72 5.739-3.16 0-5.72-2.57-5.72-5.74zm-1.43 0c0 3.962 3.2 7.174 7.15 7.174s7.15-3.212 7.15-7.174S11.848.75 7.898.75.748 3.962.748 7.924z"></path></g></svg> <svg class="wds-icon wds-global-navigation__search-toggle-icon" id="wds-icons-magnifying-glass" viewbox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><g fill-rule="evenodd"><path d="M21.747 20.524l-4.872-4.871a.864.864 0 1 0-1.222 1.222l4.871 4.872a.864.864 0 1 0 1.223-1.223z"></path><path d="M3.848 10.763a6.915 6.915 0 0 1 6.915-6.915 6.915 6.915 0 0 1 6.915 6.915 6.915 6.915 0 0 1-6.915 6.915 6.915 6.915 0 0 1-6.915-6.915zm-1.729 0a8.643 8.643 0 0 0 8.644 8.644 8.643 8.643 0 0 0 8.644-8.644 8.643 8.643 0 0 0-8.644-8.644 8.643 8.643 0 0 0-8.644 8.644z"></path></g></svg> <span class="wds-global-navigation__search-toggle-text">
    				Search			</span>
    </div>
    <div class="wds-dropdown__toggle wds-global-navigation__search-input-wrapper">
    <input autocomplete="off" class="wds-global-navigation__search-input" data-suggestions-param-name="query" data-suggestions-tracking-label="search-suggestion" data-suggestions-url="//lyrics.wikia.com/index.php?action=ajax&amp;rs=getLinkSuggest&amp;format=json" name="query" placeholder="Search LyricWiki..." type="search"/>
    <button class="wds-button wds-is-text wds-global-navigation__search-close" type="button">
    <svg class="wds-icon wds-icon-tiny wds-global-navigation__search-close-icon" id="wds-icons-plus-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M11 5H7V1a1 1 0 1 0-2 0v4H1a1 1 0 1 0 0 2h4v4a1 1 0 1 0 2 0V7h4a1 1 0 1 0 0-2" fill-rule="evenodd"></path></svg> </button>
    <button class="wds-button wds-global-navigation__search-submit" data-tracking-label="search">
    <svg class="wds-icon wds-icon-small wds-global-navigation__search-submit-icon" id="wds-icons-arrow-small" viewbox="0 0 18 18" xmlns="http://www.w3.org/2000/svg"><path d="M16 7.994H4.419l3.29-3.287a1 1 0 1 0-1.415-1.414l-5 4.997a.998.998 0 0 0-.002 1.412l4.996 5.004a.997.997 0 0 0 1.414.002.998.998 0 0 0 .002-1.414l-3.295-3.3h11.59a1 1 0 1 0 0-2" fill-rule="evenodd"></path></svg> </button>
    </div>
    </div>
    </form>
    <div class="wds-dropdown wds-global-navigation__user-menu wds-has-dark-shadow wds-global-navigation__user-anon">
    <div class="wds-dropdown__toggle">
    <div class="wds-avatar">
    <div alt="" class="wds-avatar__inner-border" title=""></div>
    <svg class="wds-avatar__image" id="wds-avatar-icon-user" viewbox="6 6 138 138" xmlns="http://www.w3.org/2000/svg"><path d="M75 76.667c11.03 0 20-8.97 20-20V50c0-11.03-8.97-20-20-20s-20 8.97-20 20v6.667c0 11.03 8.97 20 20 20zm-16.667 6.666C45.467 83.333 35 93.8 35 106.667c0 1.84-.874 23.546.966 23.546 0 0 22.608 12.983 35.606 13.978 12.998.995 42.383-8.007 42.383-8.007 1.84 0 1.045-27.677 1.045-29.517 0-12.867-10.467-23.334-23.333-23.334H58.333z"></path></svg> </div>
    <svg class="wds-icon wds-icon-tiny wds-dropdown__toggle-chevron" id="wds-icons-dropdown-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6 9l4-5H2" fill-rule="evenodd"></path></svg> </div>
    <div class="wds-dropdown__content wds-is-right-aligned">
    <ul class="wds-list wds-has-lines-between">
    <li>
    <a class="wds-button wds-is-full-width" data-tracking-label="account.sign-in" href="https://www.wikia.com/signin?redirect=http%3A%2F%2Flyrics.wikia.com%2Fwiki%2FMissy_Elliott%3ALose_Control" rel="nofollow">
    					Sign In				</a>
    </li>
    <li>
    <div class="wds-global-navigation__user-menu-dropdown-caption">Don't have an account?</div>
    <a class="wds-button wds-is-full-width wds-is-secondary" data-tracking-label="account.register" href="https://www.wikia.com/register?redirect=http%3A%2F%2Flyrics.wikia.com%2Fwiki%2FMissy_Elliott%3ALose_Control" rel="nofollow">
    					Register				</a>
    </li>
    </ul>
    </div>
    </div>
    <div class="wds-global-navigation__start-a-wiki">
    <a class="wds-button wds-is-secondary wds-global-navigation__link-button" data-tracking-label="start-a-wiki" href="//community.wikia.com/wiki/Special:CreateNewWiki">
    	Start a Wiki</a>
    </div>
    </div>
    </div>
    </div>
    </div>
    </div>
    <div class="banner-notifications-placeholder">
    </div>
    <div class="WikiaTopAds" id="WikiaTopAds">
    <div class="WikiaTopAdsInner">
    <!-- BEGIN SLOTNAME: TOP_LEADERBOARD -->
    <div class="wikia-ad noprint default-height" id="TOP_LEADERBOARD">
    <script>
    							window.adslots2.push(["TOP_LEADERBOARD"]);
    					</script>
    </div>
    <!-- END SLOTNAME: TOP_LEADERBOARD -->
    </div>
    <!-- BEGIN SLOTNAME: INVISIBLE_SKIN -->
    <div class="wikia-ad noprint default-height" id="INVISIBLE_SKIN">
    <script>
    							window.adslots2.push(["INVISIBLE_SKIN"]);
    					</script>
    </div>
    <!-- END SLOTNAME: INVISIBLE_SKIN -->
    </div>
    <div class="hidden" id="InvisibleHighImpactWrapper">
    <div class="background"></div>
    <div class="top-bar">
    <div class="label">Advertisement</div>
    <a class="close">
    <div class="close-button"></div>
    </a>
    </div>
    <div class="wikia-ad noprint" id="INVISIBLE_HIGH_IMPACT_2"></div>
    </div>
    <!-- BEGIN CONTAINER: TOP_LEADERBOARD_AB -->
    <div id="TOP_LEADERBOARD_AB">
    <script>
    			window.adslots2.push(["TOP_LEADERBOARD_AB"]);
    		</script>
    </div>
    <!-- END CONTAINER: TOP_LEADERBOARD_AB -->
    <header class="wds-community-header">
    <div class="wds-community-header__wordmark" data-tracking="wordmark-image">
    <a accesskey="z" href="//lyrics.wikia.com/wiki/LyricWiki">
    <img alt="LyricWiki" height="65" src="https://vignette.wikia.nocookie.net/lyricwiki/images/8/89/Wiki-wordmark.png/revision/latest?cb=20171025141541" width="250"/>
    </a>
    </div>
    <div class="wds-community-header__top-container">
    <div class="wds-community-header__sitename" data-tracking="sitename">
    <a href="//lyrics.wikia.com/wiki/LyricWiki">LyricWiki</a>
    </div>
    <div class="wds-community-header__counter" data-tracking="counter">
    <span class="wds-community-header__counter-value">2,030,204</span>
    <span class="wds-community-header__counter-label">Pages</span>
    </div>
    <div class="wds-community-header__wiki-buttons wds-button-group">
    <a class="wds-button wds-is-secondary createpage" data-tracking="add-new-page" href="/wiki/Special:CreatePage" title="Add new page">
    <svg class="wds-icon wds-icon-small" id="wds-icons-article-small" viewbox="0 0 18 18" xmlns="http://www.w3.org/2000/svg"><path d="M13 5.5a.5.5 0 0 0-.5-.5h-7a.5.5 0 0 0 0 1h7a.5.5 0 0 0 .5-.5zm0 3a.5.5 0 0 0-.5-.5h-7a.5.5 0 0 0 0 1h7a.5.5 0 0 0 .5-.5zm-5 3a.5.5 0 0 0-.5-.5h-2a.5.5 0 0 0 0 1h2a.5.5 0 0 0 .5-.5zM16 2v9h-6v6H3c-.6 0-1-.4-1-1V2c0-.6.4-1 1-1h12c.6 0 1 .4 1 1zm-4 11h4l-4 4v-4z" fill-rule="evenodd"></path></svg> <span>Add new page</span>
    </a>
    </div>
    </div>
    <nav class="wds-community-header__local-navigation">
    <ul class="wds-tabs">
    <li class="wds-tabs__tab">
    <div class="wds-dropdown">
    <div class="wds-tabs__tab-label wds-dropdown__toggle">
    <a data-tracking="custom-level-1" href="#">
    <span>Stroll Around</span>
    </a>
    <svg class="wds-icon wds-icon-tiny wds-dropdown__toggle-chevron" id="wds-icons-dropdown-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6 9l4-5H2" fill-rule="evenodd"></path></svg> </div>
    <div class="wds-is-not-scrollable wds-dropdown__content">
    <ul class="wds-list wds-is-linked wds-has-bolded-items">
    <li>
    <a data-tracking="custom-level-2" href="/wiki/Special:Insights/popularpages">
    												Popular Pages											</a>
    </li>
    <li class="wds-dropdown-level-2">
    <a class="wds-dropdown-level-2__toggle" data-tracking="custom-level-2" href="/wiki/Special:Random">
    <span>Random Page</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown-chevron" id="wds-icons-menu-control-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6.001 10a.997.997 0 0 1-.706-.293l-5.002-5a.999.999 0 1 1 1.414-1.414L6 7.587l4.292-4.292a.999.999 0 1 1 1.414 1.414L6.708 9.707a.997.997 0 0 1-.707.293" fill-rule="evenodd"></path></svg> </a>
    <div class="wds-is-not-scrollable wds-dropdown-level-2__content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Special:RandomInCategory/Artist">Random Artist</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Special:RandomInCategory/Album">Random Album</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Special:RandomInCategory/Song">Random Song</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Special:RandomInCategory/Translation">Random Translation</a>
    </li>
    </ul>
    </div>
    </li>
    <li>
    <a data-tracking="custom-level-2" href="/wiki/LyricWiki:Song_of_the_Day">
    												Song of the Day											</a>
    </li>
    <li>
    <a data-tracking="custom-level-2" href="/wiki/LyricWiki:Top_100">
    												iTunes Top 100											</a>
    </li>
    </ul>
    </div>
    </div>
    </li>
    <li class="wds-tabs__tab">
    <div class="wds-dropdown">
    <div class="wds-tabs__tab-label wds-dropdown__toggle">
    <a data-tracking="custom-level-1" href="#">
    <span>Browse Lyrics</span>
    </a>
    <svg class="wds-icon wds-icon-tiny wds-dropdown__toggle-chevron" id="wds-icons-dropdown-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6 9l4-5H2" fill-rule="evenodd"></path></svg> </div>
    <div class="wds-is-not-scrollable wds-dropdown__content">
    <ul class="wds-list wds-is-linked wds-has-bolded-items">
    <li class="wds-dropdown-level-2">
    <a class="wds-dropdown-level-2__toggle" data-tracking="custom-level-2" href="/wiki/Category:Artists">
    <span>Artists</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown-chevron" id="wds-icons-menu-control-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6.001 10a.997.997 0 0 1-.706-.293l-5.002-5a.999.999 0 1 1 1.414-1.414L6 7.587l4.292-4.292a.999.999 0 1 1 1.414 1.414L6.708 9.707a.997.997 0 0 1-.707.293" fill-rule="evenodd"></path></svg> </a>
    <div class="wds-is-not-scrollable wds-dropdown-level-2__content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Artists_by_First_Letter">Artists by First Letter</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre">Artists by Genre</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Hometown">Artists by Hometown</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Label">Artists by Label</a>
    </li>
    </ul>
    </div>
    </li>
    <li class="wds-dropdown-level-2">
    <a class="wds-dropdown-level-2__toggle" data-tracking="custom-level-2" href="/wiki/Category:Albums">
    <span>Albums</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown-chevron" id="wds-icons-menu-control-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6.001 10a.997.997 0 0 1-.706-.293l-5.002-5a.999.999 0 1 1 1.414-1.414L6 7.587l4.292-4.292a.999.999 0 1 1 1.414 1.414L6.708 9.707a.997.997 0 0 1-.707.293" fill-rule="evenodd"></path></svg> </a>
    <div class="wds-is-not-scrollable wds-dropdown-level-2__content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Albums_by_First_Letter">Albums by First Letter</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre">Albums by Genre</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Albums_by_Release_Year">Albums by Release Year</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Compilation_Series">Compilation Series</a>
    </li>
    </ul>
    </div>
    </li>
    <li class="wds-dropdown-level-2">
    <a class="wds-dropdown-level-2__toggle" data-tracking="custom-level-2" href="/wiki/Category:Songs">
    <span>Songs</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown-chevron" id="wds-icons-menu-control-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6.001 10a.997.997 0 0 1-.706-.293l-5.002-5a.999.999 0 1 1 1.414-1.414L6 7.587l4.292-4.292a.999.999 0 1 1 1.414 1.414L6.708 9.707a.997.997 0 0 1-.707.293" fill-rule="evenodd"></path></svg> </a>
    <div class="wds-is-not-scrollable wds-dropdown-level-2__content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Songs_by_First_Letter">Songs by First Letter</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Language">Songs by Language</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Song_Translations">Translations</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Instrumental">Instrumentals</a>
    </li>
    </ul>
    </div>
    </li>
    <li class="wds-dropdown-level-2">
    <a class="wds-dropdown-level-2__toggle" data-tracking="custom-level-2" href="/wiki/Category:Genre">
    <span>Genres</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown-chevron" id="wds-icons-menu-control-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6.001 10a.997.997 0 0 1-.706-.293l-5.002-5a.999.999 0 1 1 1.414-1.414L6 7.587l4.292-4.292a.999.999 0 1 1 1.414 1.414L6.708 9.707a.997.997 0 0 1-.707.293" fill-rule="evenodd"></path></svg> </a>
    <div class="wds-is-not-scrollable wds-dropdown-level-2__content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre/Pop">Pop</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre/Rock">Rock</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre/Electronic">Electronic</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre/Hip_Hop">Hip Hop</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre/Metal">Metal</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre/Country">Country</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre/Folk">Folk</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Category:Genre">All Genres…</a>
    </li>
    </ul>
    </div>
    </li>
    <li>
    <a data-tracking="custom-level-2" href="/wiki/Category:Label">
    												Labels											</a>
    </li>
    <li>
    <a data-tracking="custom-level-2" href="/wiki/Category:Lists">
    												Lists											</a>
    </li>
    </ul>
    </div>
    </div>
    </li>
    <li class="wds-tabs__tab">
    <div class="wds-dropdown">
    <div class="wds-tabs__tab-label wds-dropdown__toggle">
    <a data-tracking="custom-level-1" href="#">
    <span>Help Out</span>
    </a>
    <svg class="wds-icon wds-icon-tiny wds-dropdown__toggle-chevron" id="wds-icons-dropdown-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6 9l4-5H2" fill-rule="evenodd"></path></svg> </div>
    <div class="wds-is-not-scrollable wds-dropdown__content">
    <ul class="wds-list wds-is-linked wds-has-bolded-items">
    <li>
    <a data-tracking="custom-level-2" href="/wiki/Help:FAQ">
    												LyricWiki FAQ											</a>
    </li>
    <li class="wds-dropdown-level-2">
    <a class="wds-dropdown-level-2__toggle" data-tracking="custom-level-2" href="/wiki/Help:Contents">
    <span>Help</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown-chevron" id="wds-icons-menu-control-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6.001 10a.997.997 0 0 1-.706-.293l-5.002-5a.999.999 0 1 1 1.414-1.414L6 7.587l4.292-4.292a.999.999 0 1 1 1.414 1.414L6.708 9.707a.997.997 0 0 1-.707.293" fill-rule="evenodd"></path></svg> </a>
    <div class="wds-is-not-scrollable wds-dropdown-level-2__content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Help:Contents/Editing">Help with Editing Pages</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/LyricWiki:Help_Desk">Ask at the Help Desk</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/LyricWiki:Job_Exchange">File an Edit Request</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/LyricWiki:Page_Names">Page Naming Policy</a>
    </li>
    </ul>
    </div>
    </li>
    <li class="wds-is-sticked-to-parent wds-dropdown-level-2">
    <a class="wds-dropdown-level-2__toggle" data-tracking="custom-level-2" href="/wiki/LyricWiki:Community_Portal">
    <span>The Community</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown-chevron" id="wds-icons-menu-control-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6.001 10a.997.997 0 0 1-.706-.293l-5.002-5a.999.999 0 1 1 1.414-1.414L6 7.587l4.292-4.292a.999.999 0 1 1 1.414 1.414L6.708 9.707a.997.997 0 0 1-.707.293" fill-rule="evenodd"></path></svg> </a>
    <div class="wds-is-not-scrollable wds-dropdown-level-2__content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="custom-level-3" href="/wiki/LyricWiki:Administrators">Find Admins</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/LyricWiki_talk:Community_Portal">Community Portal</a>
    </li>
    </ul>
    </div>
    </li>
    <li>
    <a data-tracking="custom-level-2" href="/wiki/Category:Requests_For_Edits">
    												Requests for Edits											</a>
    </li>
    </ul>
    </div>
    </div>
    </li>
    <li class="wds-tabs__tab">
    <div class="wds-dropdown">
    <div class="wds-tabs__tab-label wds-dropdown__toggle">
    <a data-tracking="custom-level-1" href="#">
    <span>Advanced</span>
    </a>
    <svg class="wds-icon wds-icon-tiny wds-dropdown__toggle-chevron" id="wds-icons-dropdown-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6 9l4-5H2" fill-rule="evenodd"></path></svg> </div>
    <div class="wds-is-not-scrollable wds-dropdown__content">
    <ul class="wds-list wds-is-linked wds-has-bolded-items">
    <li>
    <a data-tracking="custom-level-2" href="/wiki/Special:Upload">
    												Upload Art											</a>
    </li>
    <li>
    <a data-tracking="custom-level-2" href="/wiki/Help:Lyrically_Lyrics_App">
    												Lyrically Mobile App											</a>
    </li>
    <li class="wds-dropdown-level-2">
    <a class="wds-dropdown-level-2__toggle" data-tracking="custom-level-2" href="/wiki/Special:SpecialPages">
    <span>Special Pages</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown-chevron" id="wds-icons-menu-control-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6.001 10a.997.997 0 0 1-.706-.293l-5.002-5a.999.999 0 1 1 1.414-1.414L6 7.587l4.292-4.292a.999.999 0 1 1 1.414 1.414L6.708 9.707a.997.997 0 0 1-.707.293" fill-rule="evenodd"></path></svg> </a>
    <div class="wds-is-not-scrollable wds-dropdown-level-2__content">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Special:DoubleRedirects">Double Redirects</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Special:UncategorizedFiles">Uncategorized Photos</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Special:CategoryIntersection">Category Intersection Search</a>
    </li>
    <li>
    <a data-tracking="custom-level-3" href="/wiki/Special:Wikify">Wikify Tracklists</a>
    </li>
    </ul>
    </div>
    </li>
    </ul>
    </div>
    </div>
    </li>
    <li class="wds-tabs__tab">
    <div class="wds-dropdown">
    <div class="wds-tabs__tab-label wds-dropdown__toggle">
    <svg class="wds-icon-tiny wds-icon" id="wds-icons-explore-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M10.5 8.5a5.132 5.132 0 0 0-1.875-.357c-.675 0-1.35.143-1.875.357V3.143c0-.214.675-.714 1.875-.714s1.875.5 1.875.714V8.5zm-7.125-.357c-.675 0-1.35.143-1.875.357V3.143c0-.214.675-.714 1.875-.714s1.875.5 1.875.714V8.5a5.132 5.132 0 0 0-1.875-.357zM8.625 1C7.575 1 6.6 1.286 6 1.786 5.4 1.286 4.425 1 3.375 1 1.425 1 0 1.929 0 3.143v7.143c0 .428.3.714.75.714s.75-.286.75-.714c0-.215.675-.715 1.875-.715s1.875.5 1.875.715c0 .428.3.714.75.714s.75-.286.75-.714c0-.215.675-.715 1.875-.715s1.875.5 1.875.715c0 .428.3.714.75.714s.75-.286.75-.714V3.143C12 1.929 10.575 1 8.625 1z"></path></svg> <span>Explore</span>
    <svg class="wds-icon wds-icon-tiny wds-dropdown__toggle-chevron" id="wds-icons-dropdown-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6 9l4-5H2" fill-rule="evenodd"></path></svg> </div>
    <div class="wds-is-not-scrollable wds-dropdown__content">
    <ul class="wds-list wds-is-linked wds-has-bolded-items">
    <li>
    <a data-tracking="explore-activity" href="//lyrics.wikia.com/wiki/Special:WikiActivity">Wiki Activity</a>
    </li>
    <li>
    <a data-tracking="explore-random" href="//lyrics.wikia.com/wiki/Special:Random">Random page</a>
    </li>
    <li>
    <a data-tracking="explore-community" href="//lyrics.wikia.com/wiki/Special:Community">Community</a>
    </li>
    <li>
    <a data-tracking="explore-videos" href="//lyrics.wikia.com/wiki/Special:Videos">Videos</a>
    </li>
    <li>
    <a data-tracking="explore-images" href="//lyrics.wikia.com/wiki/Special:Images">Images</a>
    </li>
    </ul>
    </div>
    </div>
    </li>
    </ul>
    </nav>
    </header>
    <!-- empty onclick event needs to be applied here to ensure that wds dropdowns work correctly on ios -->
    <section class="WikiaPage V2" id="WikiaPage" onclick="">
    <div class="WikiaPageBackground" id="WikiaPageBackground"></div>
    <div class="WikiaPageContentWrapper">
    <header class="page-header" id="PageHeader">
    <div class="page-header__main">
    <div class="page-header__categories">
    <span class="page-header__categories-in" data-tracking="categories-top-in">in:</span>
    <div class="page-header__categories-links">
    <a data-tracking="categories-top-0" href="/wiki/Category:Song">Song</a>, <a data-tracking="categories-top-1" href="/wiki/Category:Green_Songs">Green Songs</a>, <a data-tracking="categories-top-2" href="/wiki/Category:Language/English">Language/English</a>, 							<div class="wds-dropdown page-header__categories-dropdown">
    <a class="wds-dropdown__toggle" data-tracking="categories-more">and 6 more</a>
    <div class="wds-dropdown__content page-header__categories-dropdown-content wds-is-left-aligned">
    <ul class="wds-list wds-is-linked">
    <li>
    <a data-tracking="categories-top-more-0" href="/wiki/Category:Songs_by_Missy_Elliott">Songs by Missy Elliott</a> </li>
    <li>
    <a data-tracking="categories-top-more-1" href="/wiki/Category:Songs_L">Songs L</a> </li>
    <li>
    <a data-tracking="categories-top-more-2" href="/wiki/Category:ITunes/Song">ITunes/Song</a> </li>
    <li>
    <a data-tracking="categories-top-more-3" href="/wiki/Category:Spotify/Song">Spotify/Song</a> </li>
    <li>
    <a data-tracking="categories-top-more-4" href="/wiki/Category:Allmusic/Song">Allmusic/Song</a> </li>
    <li>
    <a data-tracking="categories-top-more-5" href="/wiki/Category:MusicBrainz/Song">MusicBrainz/Song</a> </li>
    </ul>
    </div>
    </div>
    </div>
    </div>
    <h1 class="page-header__title">Missy Elliott:Lose Control Lyrics</h1>
    </div>
    <div class="page-header__contribution">
    <div> <!--Empty div to ensure $actionButton is always pushed to bottom of the container-->
    </div>
    <div class="page-header__contribution-buttons">
    <div class="wds-button-group">
    <a accesskey="e" class="wds-button" data-tracking="ca-edit" href="/wiki/Missy_Elliott:Lose_Control?action=edit" id="ca-edit">
    <svg class="wds-icon wds-icon-small" id="wds-icons-pencil-small" viewbox="0 0 18 18" xmlns="http://www.w3.org/2000/svg"><path d="M9.1 4.5l-7.8 7.8c-.2.2-.3.4-.3.7v3c0 .6.4 1 1 1h3c.3 0 .5-.1.7-.3l7.8-7.8-4.4-4.4zm7.6-.2l-3-3c-.4-.4-1-.4-1.4 0l-1.8 1.8 4.4 4.4 1.8-1.8c.4-.4.4-1 0-1.4z" fill-rule="evenodd"></path></svg> <span>Edit</span>
    </a>
    <div class="wds-dropdown">
    <div class="wds-button wds-dropdown__toggle">
    <svg class="wds-icon wds-icon-tiny wds-dropdown__toggle-chevron" id="wds-icons-dropdown-tiny" viewbox="0 0 12 12" xmlns="http://www.w3.org/2000/svg"><path d="M6 9l4-5H2" fill-rule="evenodd"></path></svg> </div>
    <div class="wds-dropdown__content wds-is-not-scrollable wds-is-right-aligned">
    <ul class="wds-list wds-is-linked">
    <li>
    <a accesskey="s" data-tracking="ca-ve-edit-dropdown" href="/wiki/Missy_Elliott:Lose_Control?veaction=edit" id="ca-ve-edit">
    							VisualEditor						</a>
    </li>
    <li>
    <a data-tracking="ca-history-dropdown" href="/wiki/Missy_Elliott:Lose_Control?action=history" id="ca-history">
    							History						</a>
    </li>
    <li>
    <a class="new" data-tracking="ca-talk-dropdown" href="/wiki/Talk:Missy_Elliott:Lose_Control?action=edit&amp;redlink=1" id="ca-talk">
    							Talk (0)						</a>
    </li>
    </ul>
    </div>
    </div>
    </div>
    <a class="wds-button wds-is-secondary" href="#" id="ShareEntryPoint">
    <svg class="wds-icon wds-icon-small" id="wds-icons-share" viewbox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M5 16a3.984 3.984 0 0 0 2.861-1.213l7.295 4.168A3.964 3.964 0 0 0 15 20c0 2.206 1.794 4 4 4s4-1.794 4-4-1.794-4-4-4a3.984 3.984 0 0 0-2.861 1.213l-7.295-4.168C8.935 12.71 9 12.364 9 12c0-.364-.065-.71-.156-1.045l7.295-4.168A3.984 3.984 0 0 0 19 8c2.206 0 4-1.794 4-4s-1.794-4-4-4-4 1.794-4 4c0 .364.065.71.156 1.045L7.861 9.213A3.984 3.984 0 0 0 5 8c-2.206 0-4 1.794-4 4s1.794 4 4 4" fill-rule="evenodd"></path></svg> <span>Share</span>
    </a>
    </div>
    </div>
    </header>
    <hr class="page-header__separator"/>
    <article class="WikiaMainContent" id="WikiaMainContent">
    <div class="WikiaMainContentContainer" id="WikiaMainContentContainer">
    <div class="WikiaArticle" id="WikiaArticle">
    <div class="home-top-right-ads">
    </div>
    <div class="mw-content-ltr mw-content-text" dir="ltr" id="mw-content-text" lang="en"><div id="header-icons"><div id="ranking-star-icon"><a class="image image-thumbnail link-internal" href="/wiki/Talk:Missy_Elliott:Lose_Control" title="Page rank: Green"><img alt="StarIconGreen" class="lzy lzyPlcHld " data-image-key="StarIconGreen.png" data-image-name="StarIconGreen.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/a/a2/StarIconGreen.png/revision/latest/scale-to-width-down/16?cb=20081201112533" height="15" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="StarIconGreen" class="" data-image-key="StarIconGreen.png" data-image-name="StarIconGreen.png" height="15" src="https://vignette.wikia.nocookie.net/lyricwiki/images/a/a2/StarIconGreen.png/revision/latest/scale-to-width-down/16?cb=20081201112533" width="16"/></noscript></a></div>
    <div id="song-language-icon"><a class="image image-thumbnail link-internal" href="/wiki/Category:Language/English" title="Language: English"><img alt="LangIcon" class="lzy lzyPlcHld " data-image-key="LangIcon.png" data-image-name="LangIcon.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/3/36/LangIcon.png/revision/latest?cb=20160203212231" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="22"/><noscript><img alt="LangIcon" class="" data-image-key="LangIcon.png" data-image-name="LangIcon.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/3/36/LangIcon.png/revision/latest?cb=20160203212231" width="22"/></noscript></a></div></div>
    <div id="song-header-container">
    <div id="song-header-title"><b>Lose Control</b></div>
    <p>This song is by <b><a href="/wiki/Missy_Elliott" title="Missy Elliott">Missy Elliott</a></b> and appears on the album <i><a href="/wiki/Missy_Elliott:The_Cookbook_(2005)" title="Missy Elliott:The Cookbook (2005)">The Cookbook (2005)</a></i>.
    </p>
    </div>
    <noscript><div class="gracenote-header">You must enable javascript to view this page.  This is a requirement of our licensing agreement with music Gracenote.</div><style type="text/css">.lyricbox{display:none !important;}</style></noscript>
    <div class="lyricbox">(Hot Streak - Body Work Sample)<br/>Music make you lose control,<br/>Music make you lose control.<br/><br/>(Fatman Scoop)<br/>Let's go.<br/>Hey, yeah, yeah, yeah, yeah, yeah.<br/>Here we go now, Here we go now, Here we go now, Here we go now.<br/>One time now.<br/>Misdemeanor's in the house.<br/>Ciara's in the house.<br/>Misdemeanor's in the house.<br/>Fatman Scoop, Man Scoop, Man Scoop.<br/><br/>(Verse 1: Missy Elliott)<br/>I've got a cute face,<br/>Chubby waist,<br/>Thick legs in shape.<br/>Rump shakin' both ways,<br/>Make you do a double-take.<br/>Planet Rocker,<br/>Show stopper,<br/>Flow proper,<br/>Head knocker.<br/>Beat scholar,<br/>Tail dropper.<br/>Do my thang, motherfuckers.<br/><br/>My Rolls Royce,<br/>Lamborghini,<br/>Blue madina,<br/>Always beamin'.<br/>Ragtop,<br/>Chrome pipes,<br/>Blue lights,<br/>Outta sight.<br/><br/>Long weave,<br/>Sow it in.<br/>Say it again,<br/>Sow it in.<br/>Make that money,<br/>Throw it in.<br/>Booty bouncin',<br/>Gon' head.<br/><br/>Everybody here gettin' out of control,<br/>Get your backs off the wall, cause Misdemeanor said so.<br/>Everybody,<br/>Everybody,<br/>Everybody,<br/>Everybody.<br/><br/>(Verse 2: Ciara)<br/>Well my name is Ciara,<br/>For all you fly fellas,<br/>No one can do it better.<br/>(She'll sing on A Cappella)<br/>Boy the music makes me lose control.<br/>(We gon' make ya lose control and let it go, before ya know, you gon' hit the floor)<br/><br/>(Verse 3: Missy Elliott)<br/>I rock to the beat till I'm (tired!).<br/>Walk in the club it's (fire!).<br/>Get it crunk and wired.<br/>Wave yo hands, scream (louder!).<br/>If you smoke, then fire it up.<br/>Bring the roof down and (holler!).<br/>If you tipsy, stand up.<br/>DJ, turn it (louder!).<br/><br/>Take somebody by the waist and uh!<br/>Now, throw it in their face like uh!<br/>Hypnotic, robotic, this here will rock your bodies.<br/>Take somebody by the waist and uh!<br/>Now, throw it in their face like uh!<br/>Systematic, it's static. This hit be automatic.<br/><br/>(Breakdown)<br/>(Missy Elliott)<br/>Work me. Work, work.<br/>Work me. Work, work.<br/>Work me. Work, work.<br/>Work me, Do it right.<br/>(Fatman Scoop)<br/>(Repeat x6)<br/>Hit the floor.<br/><br/>(Fatman Scoop)<br/>Now, put your back on the wall.<br/>Put your back on the wall.<br/>Put your back on the wall.<br/>Put your back on the wall.<br/>Let's go.<br/>Misdemeanor's in the house.<br/>Ciara's in the house.<br/>Misdemeanor's in the house.<br/><br/>(Hot Streak - Body Work Sample)<br/>Music make you lose control.<br/><br/>(Fatman Scoop)<br/>We on fire.<br/>We on fire.<br/>We on fire.<br/>We on fire.<br/>Now, throw it girl.<br/>Throw it girl.<br/>Throw it girl.<br/>Yeah!<br/>Now, move your arm to the left girl.<br/>Now, move your arm to the left girl.<br/>Now, move your arm to the right girl.<br/>Now, move your arm to the right girl.<br/>Let's go now, let's go now, let's go now. Whoo! Let's go.<br/>Should I bring it back right now? Now bring it back now!<br/>Whoo! Oh!<br/>Oh I see you C.<br/>Now, see I'ma, I'ma do it like C do it.<br/>Now, shake it girl.<br/>C'mon 'n' just shake it girl.<br/>C'mon 'n' let it pop right girl.<br/>C'mon 'n' let it pop right girl.<br/>Now, yo!<br/>Now, back it up girl.<br/>Back it up girl.<br/>Back it up girl.<br/>Back it up girl.<br/>Whoo! Whoo! Whoo! Yo! Yo!<br/>Bring it to the front girl.<br/>Go! Go!<br/>Bring it to the front girl.<br/>Go! Go!<br/>Bring it to the front girl.<br/>Go! Go!<br/>Bring it to the front girl.<br/>Let's go! Let's go!<div class="lyricsbreak"></div>
    </div>
    <h2 style="clear:both"><span class="mw-headline" id="External_links">External links</span></h2><div class="heading-opposite-text mobile-hidden plainlinks"><span id="sotd-link"><a class="external text" href="//lyrics.wikia.com/wiki/Special:SOTD?artist=Missy+Elliott&amp;song=Lose+Control&amp;vlink=">Nominate as Song of the Day</a></span></div><div class="plainlinks extlink"><img alt="ITunes icon" class="lzy lzyPlcHld " data-image-key="ITunes_icon.png" data-image-name="ITunes icon.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/0/06/ITunes_icon.png/revision/latest?cb=20170506180208" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="ITunes icon" class="" data-image-key="ITunes_icon.png" data-image-name="ITunes icon.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/0/06/ITunes_icon.png/revision/latest?cb=20170506180208" width="16"/></noscript> iTunes: buy <a class="external text" href="https://itunes.apple.com/us/album/73240428?i=73240226"><b>Lose Control</b></a></div><div class="plainlinks extlink"><img alt="Amazon Icon" class="lzy lzyPlcHld " data-image-key="Amazon_Icon.png" data-image-name="Amazon Icon.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/a/ab/Amazon_Icon.png/revision/latest?cb=20170103063818" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="Amazon Icon" class="" data-image-key="Amazon_Icon.png" data-image-name="Amazon Icon.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/a/ab/Amazon_Icon.png/revision/latest?cb=20170103063818" width="16"/></noscript> Amazon: search for… <a class="external text" href="https://www.amazon.com/exec/obidos/redirect?link_code=ur2&amp;tag=wikia-20&amp;camp=1789&amp;creative=9325&amp;path=external-search%3Fsearch-type=ss%26index=music%26ie=UTF8%26keyword=Missy+Elliott">Missy Elliott</a> • <a class="external text" href="https://www.amazon.com/exec/obidos/redirect?link_code=ur2&amp;tag=wikia-20&amp;camp=1789&amp;creative=9325&amp;path=external-search%3Fsearch-type=ss%26index=music%26ie=UTF8%26keyword=Missy+Elliott+The+Cookbook">The Cookbook</a> • <a class="external text" href="https://www.amazon.com/exec/obidos/redirect?link_code=ur2&amp;tag=wikia-20&amp;camp=1789&amp;creative=9325&amp;path=external-search%3Fsearch-type=ss%26index=digital-music%26ie=UTF8%26keyword=Missy+Elliott+Lose+Control">Lose Control</a></div>
    <div class="plainlinks extlink"><img alt="Kingnee - Hype Machine" class="lzy lzyPlcHld " data-image-key="Kingnee_-_Hype_Machine.png" data-image-name="Kingnee - Hype Machine.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/6/63/Kingnee_-_Hype_Machine.png/revision/latest?cb=20080702230048" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="Kingnee - Hype Machine" class="" data-image-key="Kingnee_-_Hype_Machine.png" data-image-name="Kingnee - Hype Machine.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/6/63/Kingnee_-_Hype_Machine.png/revision/latest?cb=20080702230048" width="16"/></noscript> Hype Machine: search for… <a class="external text" href="https://hypem.com/search/Missy+Elliott">Missy Elliott</a> • <a class="external text" href="https://hypem.com/search/Missy+Elliott+Lose+Control">Lose Control</a></div>
    <div class="plainlinks extlink"><img alt="Last.fm icon" class="lzy lzyPlcHld " data-image-key="Last.fm_icon.png" data-image-name="Last.fm icon.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/1/1a/Last.fm_icon.png/revision/latest?cb=20100519162230" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="Last.fm icon" class="" data-image-key="Last.fm_icon.png" data-image-name="Last.fm icon.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/1/1a/Last.fm_icon.png/revision/latest?cb=20100519162230" width="16"/></noscript> Last.fm: search for… <a class="external text" href="https://www.last.fm/music/Missy+Elliott">Missy Elliott</a> • <a class="external text" href="https://www.last.fm/music/Missy+Elliott/The+Cookbook">The Cookbook</a> • <a class="external text" href="https://www.last.fm/music/Missy+Elliott/_/Lose+Control">Lose Control</a></div>
    <div class="plainlinks extlink"><img alt="Kingnee - Pandora" class="lzy lzyPlcHld " data-image-key="Kingnee_-_Pandora.PNG" data-image-name="Kingnee - Pandora.PNG" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/9/94/Kingnee_-_Pandora.PNG/revision/latest?cb=20081007034926" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="Kingnee - Pandora" class="" data-image-key="Kingnee_-_Pandora.PNG" data-image-name="Kingnee - Pandora.PNG" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/9/94/Kingnee_-_Pandora.PNG/revision/latest?cb=20081007034926" width="16"/></noscript> Pandora: search for… <a class="external text" href="https://www.pandora.com/backstage?type=artist&amp;q=Missy+Elliott">Missy Elliott</a> • <a class="external text" href="https://www.pandora.com/backstage?type=song&amp;q=Lose+Control">Lose Control</a></div><div class="plainlinks extlink"><img alt="Wikipedia16" class="lzy lzyPlcHld " data-image-key="Wikipedia16.png" data-image-name="Wikipedia16.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/3/3f/Wikipedia16.png/revision/latest?cb=20160716072648" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="Wikipedia16" class="" data-image-key="Wikipedia16.png" data-image-name="Wikipedia16.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/3/3f/Wikipedia16.png/revision/latest?cb=20160716072648" width="16"/></noscript> Wikipedia: search for… <a class="external text" href="https://en.wikipedia.org/wiki/Special:Search?search=Missy+Elliott">Missy Elliott</a> • <a class="external text" href="https://en.wikipedia.org/wiki/Special:Search?search=The+Cookbook+%28album%29">The Cookbook</a> • <a class="external text" href="https://en.wikipedia.org/wiki/Special:Search?search=Lose+Control+%28song%29">Lose Control</a></div><div class="plainlinks extlink"><img alt="Spotify 16" class="lzy lzyPlcHld " data-image-key="Spotify_16.png" data-image-name="Spotify 16.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/b/b5/Spotify_16.png/revision/latest?cb=20141120181139" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="Spotify 16" class="" data-image-key="Spotify_16.png" data-image-name="Spotify 16.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/b/b5/Spotify_16.png/revision/latest?cb=20141120181139" width="16"/></noscript> Spotify: <a class="external text" href="https://open.spotify.com/track/26OjArXW0vnGTtJCXVM9bG"><b>Lose Control</b></a></div><div class="plainlinks extlink"><img alt="AllMusic i" class="lzy lzyPlcHld " data-image-key="AllMusic_i.png" data-image-name="AllMusic i.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/c/cb/AllMusic_i.png/revision/latest?cb=20131016161709" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="AllMusic i" class="" data-image-key="AllMusic_i.png" data-image-name="AllMusic i.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/c/cb/AllMusic_i.png/revision/latest?cb=20131016161709" width="16"/></noscript> AllMusic: <a class="external text" href="https://www.allmusic.com/song/mt0014467313"><b>Lose Control</b></a></div><div class="plainlinks extlink"><img alt="Kingnee - Musicbrainz" class="lzy lzyPlcHld " data-image-key="Kingnee_-_Musicbrainz.png" data-image-name="Kingnee - Musicbrainz.png" data-src="https://vignette.wikia.nocookie.net/lyricwiki/images/9/91/Kingnee_-_Musicbrainz.png/revision/latest?cb=20170326061024" height="16" onload="if(typeof ImgLzy==='object'){ImgLzy.load(this)}" src="data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D" width="16"/><noscript><img alt="Kingnee - Musicbrainz" class="" data-image-key="Kingnee_-_Musicbrainz.png" data-image-name="Kingnee - Musicbrainz.png" height="16" src="https://vignette.wikia.nocookie.net/lyricwiki/images/9/91/Kingnee_-_Musicbrainz.png/revision/latest?cb=20170326061024" width="16"/></noscript> MusicBrainz: <a class="external text" href="https://musicbrainz.org/recording/775b7a9f-93d3-4b60-a9a8-56740b0851d5"><b>Lose Control</b></a></div>
    <!-- 
    NewPP limit report
    Preprocessor node count: 1008/300000
    Post‐expand include size: 12910/2097152 bytes
    Template argument size: 1722/2097152 bytes
    Expensive parser function count: 1/100
    Lua engine used: Scribunto_LuaSandboxEngine
    Lua time usage: 0.012s
    Lua memory usage: 1.11 MB
    ExtLoops count: 2/100
    -->
    <!-- Saved in parser cache with key lyricwiki:pcache:idhash:275575-0!*!0!*!*!2!* -->
    <noscript><link href="https://slot1-images.wikia.nocookie.net/__cb7800027800012/common/extensions/wikia/ImageLazyLoad/css/ImageLazyLoadNoScript.css" rel="stylesheet"/></noscript></div><div class="printfooter">
    Retrieved from "<a href="http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control?oldid=30818251">http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control?oldid=30818251</a>"</div>
    </div>
    <nav class="article-categories CategorySelect articlePage userCanEdit" id="articleCategories">
    <div class="container">
    <div class="special-categories"><a class="categoriesLink" href="/wiki/Special:Categories" rel="nofollow" title="Special:Categories">Categories</a>:</div>
    <ul class="categories">
    <li class="category normal" data-name="Song" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:Song" title="Category:Song">Song</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="category normal" data-name="Green Songs" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:Green_Songs" title="Category:Green Songs">Green Songs</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="category normal" data-name="Language/English" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:Language/English" title="Category:Language/English">Language/English</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="category normal" data-name="Songs by Missy Elliott" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:Songs_by_Missy_Elliott" title="Category:Songs by Missy Elliott">Songs by Missy Elliott</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="category normal" data-name="Songs L" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:Songs_L" title="Category:Songs L">Songs L</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="category normal" data-name="ITunes/Song" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:ITunes/Song" title="Category:ITunes/Song">ITunes/Song</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="category normal" data-name="Spotify/Song" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:Spotify/Song" title="Category:Spotify/Song">Spotify/Song</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="category normal" data-name="Allmusic/Song" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:Allmusic/Song" title="Category:Allmusic/Song">Allmusic/Song</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="category normal" data-name="MusicBrainz/Song" data-namespace="" data-outertag="" data-sortkey="" data-type="normal">
    <span class="name"><a href="/wiki/Category:MusicBrainz/Song" title="Category:MusicBrainz/Song">MusicBrainz/Song</a></span>
    <ul class="toolbar">
    <li class="tool editCategory sprite-small edit" title=""></li>
    <li class="tool removeCategory sprite-small delete" title=""></li>
    </ul>
    </li>
    <li class="last">
    <button class="wikia-button secondary add" id="CategorySelectAdd" type="button">Add category</button>
    <input autocomplete="off" class="input" id="CategorySelectInput" name="CategorySelectInput" placeholder="Add category..." type="text"/> </li>
    </ul>
    </div>
    <div class="toolbar">
    <button class="wikia-button secondary cancel" id="CategorySelectCancel" type="button">Cancel</button>
    <button class="wikia-button save" disabled="disabled" id="CategorySelectSave" type="button">Save</button>
    </div>
    </nav>
    </div>
    </article><!-- WikiaMainContent -->
    <div class="WikiaRail" id="WikiaRailWrapper">
    <div class="wikia-rail-inner" id="WikiaRail">
    <div id="top-right-boxad-wrapper">
    <!-- BEGIN SLOTNAME: TOP_RIGHT_BOXAD -->
    <div class="wikia-ad noprint default-height" id="TOP_RIGHT_BOXAD">
    <script>
    							window.adslots2.push(["TOP_RIGHT_BOXAD"]);
    					</script>
    </div>
    <!-- END SLOTNAME: TOP_RIGHT_BOXAD -->
    </div>
    <div class="loading"></div>
    </div>
    </div>
    <footer class="WikiaFooter notoolbar" id="WikiaFooter">
    <div id="bottomLeaderboardWrapper">
    <!-- BEGIN SLOTNAME: BOTTOM_LEADERBOARD -->
    <div class="wikia-ad noprint default-height" id="BOTTOM_LEADERBOARD">
    </div>
    <!-- END SLOTNAME: BOTTOM_LEADERBOARD -->
    </div>
    <div class="mcf-en" data-number-of-ns-articles="11" data-number-of-wiki-articles="9" id="mixed-content-footer">
    <div class="mcf-content">
    <h1 class="mcf-header">Fan Feed		</h1>
    <div class="mcf-mosaic">
    <div class="mcf-column">
    <div class="mcf-card-ns-placeholder" data-tracking="card-1"></div>
    <div class="mcf-card-ns-placeholder" data-tracking="card-4"></div>
    <div class="mcf-card-wiki-placeholder" data-tracking="card-7"></div>
    <div class="mcf-card-ns-placeholder mcf-card-tall" data-tracking="card-10"></div>
    <div class="mcf-card-wiki-placeholder" data-tracking="card-13"></div>
    <div class="mcf-card-ns-placeholder" data-tracking="card-16"></div>
    <div class="mcf-card-wiki-placeholder mcf-card-tall" data-tracking="card-19"></div>
    </div>
    <div class="mcf-column">
    <div class="mcf-card-ns-placeholder" data-tracking="card-2"></div>
    <div class="mcf-card-ns-placeholder" data-tracking="card-5"></div>
    <div class="mcf-card-ns-placeholder" data-tracking="card-8"></div>
    <div class="mcf-card-wiki-placeholder" data-tracking="card-11"></div>
    <div class="mcf-card-ns-placeholder" data-tracking="card-14"></div>
    <div class="mcf-card-wiki-placeholder" data-tracking="card-17"></div>
    <div class="mcf-card mcf-card-related-wikis">
    <header class="mcf-card-related-wikis__header">
    		Explore Wikis
    	</header>
    <ul class="mcf-card-related-wikis__list">
    <li class="mcf-card-related-wikis__item">
    <a class="mcf-card-related-wikis__item-link" data-tracking="explore-wikis-1" href="https://inheritance.fandom.com/wiki/Main_Page" title="Inheritance Wiki">
    <img class="mcf-card-related-wikis__item-image" src="https://vignette.wikia.nocookie.net/inheritance/images/f/fd/Wikia-Visualization-Add-2.png/revision/latest/zoom-crop/width/100/height/56?cb=20121216234452"/>
    <div>
    <span class="mcf-card-related-wikis__title">Inheritance Wiki</span>
    </div>
    </a>
    </li>
    <li class="mcf-card-related-wikis__item">
    <a class="mcf-card-related-wikis__item-link" data-tracking="explore-wikis-2" href="https://mini-4wd.wikia.com/wiki/Mini_4wd_Wiki" title="Mini 4WD Wiki">
    <img class="mcf-card-related-wikis__item-image" src="https://vignette.wikia.nocookie.net/mini-4wd/images/0/0e/Spotlight2.png/revision/latest/zoom-crop/width/100/height/56?cb=20181201050952"/>
    <div>
    <span class="mcf-card-related-wikis__title">Mini 4WD Wiki</span>
    </div>
    </a>
    </li>
    <li class="mcf-card-related-wikis__item">
    <a class="mcf-card-related-wikis__item-link" data-tracking="explore-wikis-3" href="https://glitchtale.wikia.com/wiki/Glitchtale_Wikia" title="Glitchtale Wiki">
    <img class="mcf-card-related-wikis__item-image" src="https://vignette.wikia.nocookie.net/glitchtale/images/c/c2/Glitchtale_Wallpaper.png/revision/latest/zoom-crop/width/100/height/56?cb=20180626082439"/>
    <div>
    <span class="mcf-card-related-wikis__title">Glitchtale Wiki</span>
    </div>
    </a>
    </li>
    </ul>
    </div>
    </div>
    <div class="mcf-column">
    <div class="mcf-card-wiki-placeholder" data-tracking="card-3"></div>
    <div class="mcf-card-wiki-placeholder mcf-card-tall" data-tracking="card-6"></div>
    <div class="mcf-card-wiki-placeholder" data-tracking="card-9"></div>
    <div class="mcf-card-ns-placeholder" data-tracking="card-12"></div>
    <div class="mcf-card-wiki-placeholder mcf-card-tall" data-tracking="card-15"></div>
    <div class="mcf-card-ns-placeholder" data-tracking="card-18"></div>
    <div class="mcf-card-ns-placeholder" data-tracking="card-21"></div>
    </div>
    </div>
    </div>
    </div>
    </footer>
    </div>
    </section><!--WikiaPage-->
    <footer class="wds-global-footer">
    <h2 class="wds-global-footer__header">
    <a data-tracking-label="logo" href="//fandom.wikia.com/" title="Fandom powered by Wikia">
    <svg alt="Fandom powered by Wikia" class="wds-global-footer__header-logo" id="wds-company-logo-fandom-white" viewbox="0 0 164 35" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd"><path d="M32.003 16.524c0 .288-.115.564-.32.768L18.3 30.712c-.226.224-.454.324-.738.324-.292 0-.55-.11-.77-.325l-.943-.886a.41.41 0 0 1-.01-.59l15.45-15.46c.262-.263.716-.078.716.29v2.46zm-17.167 10.12l-.766.685a.642.642 0 0 1-.872-.02L3.01 17.362c-.257-.25-.4-.593-.4-.95v-1.858c0-.67.816-1.007 1.298-.536l10.814 10.56c.188.187.505.57.505 1.033 0 .296-.068.715-.39 1.035zM5.73 7.395L9.236 3.93a.421.421 0 0 1 .592 0l11.736 11.603a3.158 3.158 0 0 1 0 4.5l-3.503 3.462a.423.423 0 0 1-.59 0L5.732 11.89a3.132 3.132 0 0 1-.937-2.25c0-.85.332-1.65.935-2.246zm13.89 1.982l3.662-3.62a3.232 3.232 0 0 1 2.737-.897c.722.098 1.378.47 1.893.978l3.708 3.667a.41.41 0 0 1 0 .585l-5.64 5.576a.419.419 0 0 1-.59 0l-5.77-5.704a.411.411 0 0 1 0-.585zm14.56-.687L26.014.475a.869.869 0 0 0-1.228-.002L18.307 6.94c-.5.5-1.316.5-1.82.004l-6.48-6.4A.87.87 0 0 0 8.793.542L.447 8.67C.16 8.95 0 9.33 0 9.727v7.7c0 .392.158.77.44 1.048l16.263 16.072a.87.87 0 0 0 1.22 0l16.25-16.073c.28-.278.438-.655.438-1.048V9.73c0-.39-.153-.763-.43-1.04z" fill="#00D6D6"></path><path d="M62.852 20.51l2.58-6.716a.468.468 0 0 1 .87 0l2.58 6.717h-6.03zm5.856-12.428c-.184-.48-.65-.8-1.17-.8h-3.342c-.52 0-.986.32-1.17.8l-7.083 18.5c-.21.552.2 1.14.796 1.14h2.753c.353 0 .67-.215.796-.542l.738-1.922a.849.849 0 0 1 .795-.542h8.088a.85.85 0 0 1 .796.542l.74 1.922c.125.327.44.543.795.543h2.754a.843.843 0 0 0 .796-1.14l-7.082-18.5zm93.504-.8h-2.715a1.86 1.86 0 0 0-1.677 1.047l-5.393 11.162-5.393-11.163a1.858 1.858 0 0 0-1.677-1.047h-2.715a.889.889 0 0 0-.893.883V26.84c0 .487.4.883.892.883h2.608a.889.889 0 0 0 .893-.883v-9.686l4.945 10.072c.15.304.46.497.803.497h1.073a.893.893 0 0 0 .803-.497l4.945-10.072v9.686c0 .487.4.883.894.883h2.608a.889.889 0 0 0 .893-.883V8.166c0-.487-.4-.883-.893-.883zm-106.972 8.8h-8.63V11.49h10.918a.88.88 0 0 0 .83-.578l.888-2.464a.872.872 0 0 0-.83-1.163h-15.18c-.486 0-.88.39-.88.87v18.7c0 .48.394.87.88.87h2.492c.486 0 .88-.39.88-.87V20.29h7.743a.88.88 0 0 0 .83-.578l.89-2.464a.872.872 0 0 0-.83-1.163zm51.76 7.61h-3.615V11.315H107c3.828 0 6.41 2.517 6.41 6.188 0 3.672-2.582 6.19-6.41 6.19zm-.124-16.41h-7.128c-.486 0-.88.39-.88.872v18.698c0 .48.394.87.88.87h7.128c6.453 0 10.912-4.44 10.912-10.16v-.117c0-5.72-4.46-10.162-10.912-10.162zm-11.947.03h-2.642a.87.87 0 0 0-.876.866v12.36l-8.755-12.72a1.242 1.242 0 0 0-1.023-.535H78.32a.873.873 0 0 0-.876.867v18.706c0 .48.393.867.877.867h2.64a.872.872 0 0 0 .878-.867V14.71l8.608 12.478c.23.334.613.535 1.022.535h3.46a.872.872 0 0 0 .877-.867V8.178a.87.87 0 0 0-.876-.867zm40.71 10.3c0 3.323-2.712 6.016-6.056 6.016-3.345 0-6.056-2.693-6.056-6.015v-.22c0-3.322 2.71-6.015 6.056-6.015 3.344 0 6.055 2.693 6.055 6.015v.22zm-6.056-10.44c-5.694 0-10.31 4.576-10.31 10.22v.22c0 5.646 4.616 10.22 10.31 10.22 5.693 0 10.308-4.574 10.308-10.22v-.22c0-5.644-4.615-10.22-10.308-10.22z" fill="#FFF"></path></g></svg> </a>
    </h2>
    <div class="wds-global-footer__main">
    <div class="wds-global-footer__fandom-sections">
    <section class="wds-global-footer__fandom-section wds-is-fandom-overview">
    <h3 class="wds-global-footer__section-header">Explore</h3>
    <ul class="wds-global-footer__links-list">
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link wds-is-games" data-tracking-label="fandom-overview.games" href="//fandom.wikia.com/games">
    <div>Games</div>
    </a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link wds-is-movies" data-tracking-label="fandom-overview.movies" href="//fandom.wikia.com/movies">
    <div>Movies</div>
    </a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link wds-is-tv" data-tracking-label="fandom-overview.tv" href="//fandom.wikia.com/tv">
    <div>TV</div>
    </a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link wds-is-explore-wikis" data-tracking-label="fandom-overview.explore-wikis" href="//fandom.wikia.com/explore">
    <div>Wikis</div>
    </a>
    </li>
    </ul>
    </section>
    <section class="wds-global-footer__fandom-section wds-is-follow-us">
    <h3 class="wds-global-footer__section-header">Follow Us</h3>
    <ul class="wds-global-footer__links-list">
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="follow-us.facebook" href="https://www.facebook.com/getfandom">
    <svg class="wds-global-footer__image wds-icon" id="wds-icons-facebook" viewbox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M16.762 5.432h-1.786c-1.428 0-1.666.71-1.666 1.657v2.248h3.452l-.357 3.55h-2.857V22H9.976v-9.112H7v-3.55h2.976V6.733C9.976 3.775 11.762 2 14.381 2c1.19 0 2.262.118 2.619.118v3.314h-.238z"></path></svg></a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="follow-us.twitter" href="https://twitter.com/getfandom">
    <svg class="wds-global-footer__image wds-icon" id="wds-icons-twitter" viewbox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20.681 7.328v.577c0 5.915-4.486 12.695-12.735 12.695-2.605 0-4.92-.721-6.946-2.02.434 0 .724.145 1.158.145 2.026 0 4.052-.722 5.644-1.876-1.882 0-3.618-1.298-4.197-3.174.29 0 .579.145.868.145.434 0 .434 0 1.013-.145-2.17-.432-4.052-2.308-4.052-4.472 0 .433 1.592.433 2.316.577-1.158-.865-1.882-2.164-1.882-3.75 0-.866.29-1.587.724-2.309 2.17 2.741 5.644 4.472 9.261 4.761-.144-.433-.144-.721-.144-1.01C11.709 5.02 13.735 3 16.195 3c1.302 0 2.46.433 3.328 1.443 1.013-.289 1.882-.577 2.75-1.154-.434 1.154-1.158 1.875-1.881 2.452a13.73 13.73 0 0 0 2.604-.721c-.723.865-1.447 1.73-2.315 2.308z"></path></svg></a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="follow-us.youtube" href="https://www.youtube.com/channel/UC988qTQImTjO7lUdPfYabgQ">
    <svg class="wds-global-footer__image wds-icon" id="wds-icons-youtube" viewbox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M23.8 7.6s-.2-1.7-1-2.4c-.9-1-1.9-1-2.4-1C17 4 12 4 12 4s-5 0-8.4.2c-.5.1-1.5.1-2.4 1-.7.7-1 2.4-1 2.4S0 9.5 0 11.5v1.8c0 1.9.2 3.9.2 3.9s.2 1.7 1 2.4c.9 1 2.1.9 2.6 1 1.9.2 8.2.2 8.2.2s5 0 8.4-.3c.5-.1 1.5-.1 2.4-1 .7-.7 1-2.4 1-2.4s.2-1.9.2-3.9v-1.8c0-1.9-.2-3.8-.2-3.8zM9.5 15.5V8.8l6.5 3.4-6.5 3.3z"></path></svg></a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="follow-us.instagram" href="https://www.instagram.com/getfandom/">
    <svg class="wds-global-footer__image wds-icon" id="wds-icons-instagram" viewbox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><g fill-rule="evenodd"><path d="M12 2.162c3.204 0 3.584.012 4.849.07 1.366.062 2.633.336 3.608 1.311.975.975 1.249 2.242 1.311 3.608.058 1.265.07 1.645.07 4.849s-.012 3.584-.07 4.849c-.062 1.366-.336 2.633-1.311 3.608-.975.975-2.242 1.249-3.608 1.311-1.265.058-1.645.07-4.849.07s-3.584-.012-4.849-.07c-1.366-.062-2.633-.336-3.608-1.311-.975-.975-1.249-2.242-1.311-3.608-.058-1.265-.07-1.645-.07-4.849s.012-3.584.07-4.849c.062-1.366.336-2.633 1.311-3.608.975-.975 2.242-1.249 3.608-1.311 1.265-.058 1.645-.07 4.849-.07zM12 0C8.741 0 8.332.014 7.052.072c-1.95.089-3.663.567-5.038 1.942C.639 3.389.161 5.102.072 7.052.014 8.332 0 8.741 0 12c0 3.259.014 3.668.072 4.948.089 1.95.567 3.663 1.942 5.038 1.375 1.375 3.088 1.853 5.038 1.942C8.332 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 1.95-.089 3.663-.567 5.038-1.942 1.375-1.375 1.853-3.088 1.942-5.038.058-1.28.072-1.689.072-4.948 0-3.259-.014-3.668-.072-4.948-.089-1.95-.567-3.663-1.942-5.038C20.611.639 18.898.161 16.948.072 15.668.014 15.259 0 12 0z"></path><path d="M12 5.838a6.162 6.162 0 1 0 0 12.324 6.162 6.162 0 0 0 0-12.324zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8z"></path><circle cx="18.406" cy="5.594" r="1.44"></circle></g></svg></a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="follow-us.linkedin" href="https://www.linkedin.com/company/157252">
    <svg class="wds-global-footer__image wds-icon" id="wds-icons-linkedin" viewbox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M.351 24h4.982V7.93H.351zM18.035 7.509c-2.386 0-4.07 1.333-4.702 2.596h-.07V7.93H8.491V24h4.983v-7.93c0-2.105.421-4.14 3.017-4.14 2.597 0 2.597 2.386 2.597 4.28V24H24v-8.842c0-4.351-.912-7.65-5.965-7.65M2.877 0A2.845 2.845 0 0 0 0 2.877a2.845 2.845 0 0 0 2.877 2.877c1.614 0 2.877-1.333 2.877-2.877A2.845 2.845 0 0 0 2.877 0"></path></svg></a>
    </li>
    </ul>
    </section>
    </div>
    <div class="wds-global-footer__wikia-sections">
    <section class="wds-global-footer__wikia-section wds-is-company-overview">
    <h3 class="wds-global-footer__section-header">Overview</h3>
    <ul class="wds-global-footer__links-list">
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="company-overview.about" href="//www.wikia.com/about">
    	About</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="company-overview.careers" href="https://careers.wikia.com">
    	Careers</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="company-overview.press" href="//fandom.wikia.com/press">
    	Press</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="company-overview.contact" href="//fandom.wikia.com/about#contact">
    	Contact</a>
    </li>
    </ul>
    </section>
    <section class="wds-global-footer__wikia-section wds-is-site-overview">
    <ul class="wds-global-footer__links-list">
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="site-overview.terms-of-use" href="//www.wikia.com/Terms_of_use">
    	Terms of Use</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="site-overview.privacy-policy" href="//www.wikia.com/Privacy_Policy">
    	Privacy Policy</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="site-overview.global-sitemap" href="//www.wikia.com/Sitemap">
    	Global Sitemap</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="site-overview.local-sitemap" href="/wiki/Local_Sitemap">
    	Local Sitemap</a>
    </li>
    </ul>
    </section>
    <section class="wds-global-footer__wikia-section wds-is-community">
    <h3 class="wds-global-footer__section-header">Community</h3>
    <ul class="wds-global-footer__links-list">
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="community.community-central" href="//community.wikia.com/wiki/Community_Central">
    	Community Central</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="community.support" href="//community.wikia.com/wiki/Special:Contact">
    	Support</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="community.fan-contributor" href="//fandom.wikia.com/fan-contributor">
    	Fan Contributor Program</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="community.wam" href="//community.wikia.com/wiki/WAM">
    	WAM Score</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="community.help" href="//community.wikia.com/wiki/Help:Contents">
    	Help</a>
    </li>
    </ul>
    </section>
    <section class="wds-global-footer__wikia-section wds-is-create-wiki">
    <span class="wds-global-footer__section-description">Can't find a community you love? Create your own and start something epic.</span>
    <ul class="wds-global-footer__links-list">
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="start-a-wiki" href="//community.wikia.com/wiki/Special:CreateNewWiki">
    	Start a wiki</a>
    </li>
    </ul>
    </section>
    <section class="wds-global-footer__wikia-section wds-is-community-apps">
    <h3 class="wds-global-footer__section-header">The FANDOM App</h3>
    <span class="wds-global-footer__section-description">Take your favorite fandoms with you and never miss a beat</span>
    <ul class="wds-global-footer__links-list">
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="community-apps.app-store" href="https://itunes.apple.com/us/app/fandom-videos-news-reviews/id1230063803?ls=1&amp;mt=8">
    <svg class="wds-global-footer__image wds-icon" id="wds-company-store-appstore" viewbox="0 0 119 35" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd"><path d="M114.766 35H4.17C1.87 35 0 33.138 0 30.859V4.135C0 1.855 1.87 0 4.169 0h110.597C117.063 0 119 1.855 119 4.135V30.86c0 2.279-1.937 4.141-4.234 4.141" fill="#A9AAA9"></path><path d="M118.147 30.86c0 1.851-1.511 3.35-3.38 3.35H4.17c-1.87 0-3.385-1.498-3.385-3.35V4.134C.785 2.284 2.3.78 4.169.78h110.597c1.87 0 3.38 1.505 3.38 3.355V30.86" fill="#0A0B09"></path><path d="M26.557 17.311c-.025-2.82 2.327-4.192 2.434-4.257-1.332-1.928-3.396-2.19-4.122-2.211-1.734-.181-3.416 1.03-4.299 1.03-.9 0-2.262-1.012-3.727-.983-1.885.03-3.65 1.113-4.619 2.797-1.997 3.432-.507 8.477 1.406 11.251.958 1.36 2.076 2.877 3.54 2.823 1.432-.06 1.967-.907 3.696-.907 1.713 0 2.216.907 3.709.873 1.537-.025 2.505-1.365 3.429-2.736 1.106-1.558 1.55-3.092 1.568-3.171-.036-.012-2.986-1.129-3.015-4.509m-2.82-8.293c.77-.957 1.296-2.258 1.15-3.578-1.115.049-2.51.765-3.312 1.7-.71.825-1.345 2.176-1.181 3.447 1.253.092 2.539-.628 3.343-1.57M43.858 22.71l-.992-3.04c-.104-.31-.301-1.042-.592-2.194h-.034a84.012 84.012 0 0 1-.557 2.195l-.974 3.04h3.15zm3.43 4.856h-2.003l-1.096-3.42h-3.811l-1.045 3.42h-1.948l3.776-11.644h2.332l3.794 11.644zm7.796-4.233c0-.817-.185-1.491-.557-2.021-.407-.553-.951-.83-1.636-.83a1.96 1.96 0 0 0-1.262.459 2.115 2.115 0 0 0-.74 1.2 2.42 2.42 0 0 0-.087.57v1.4c0 .61.19 1.126.566 1.546.378.42.868.63 1.47.63.708 0 1.26-.27 1.654-.811.394-.542.592-1.256.592-2.143zm1.914-.068c0 1.427-.388 2.556-1.166 3.385-.696.738-1.56 1.106-2.593 1.106-1.113 0-1.914-.397-2.401-1.193h-.036v4.423h-1.879v-9.052c0-.898-.024-1.82-.07-2.765h1.653l.105 1.331h.035c.626-1.002 1.578-1.503 2.854-1.503.998 0 1.83.391 2.498 1.174.667.784 1 1.815 1 3.094zm7.815.068c0-.817-.186-1.491-.557-2.021-.407-.553-.952-.83-1.636-.83-.465 0-.885.154-1.263.459a2.123 2.123 0 0 0-.738 1.2c-.058.231-.088.42-.088.57v1.4c0 .61.189 1.126.564 1.546.378.42.869.63 1.473.63.707 0 1.258-.27 1.652-.811.395-.542.593-1.256.593-2.143zm1.914-.068c0 1.427-.388 2.556-1.167 3.385-.695.738-1.56 1.106-2.592 1.106-1.114 0-1.915-.397-2.401-1.193h-.036v4.423h-1.879v-9.052c0-.898-.024-1.82-.07-2.765h1.654l.104 1.331h.035c.626-1.002 1.576-1.503 2.854-1.503.997 0 1.83.391 2.498 1.174.666.784 1 1.815 1 3.094zm10.878 1.036c0 .99-.347 1.796-1.042 2.418-.764.68-1.828 1.02-3.196 1.02-1.26 0-2.274-.242-3.04-.726l.436-1.555a5.18 5.18 0 0 0 2.716.743c.708 0 1.26-.158 1.655-.476.395-.317.59-.741.59-1.272 0-.473-.16-.87-.487-1.194-.323-.323-.864-.623-1.617-.9-2.054-.76-3.081-1.874-3.081-3.34 0-.956.36-1.741 1.08-2.353.717-.611 1.674-.917 2.871-.917 1.068 0 1.955.184 2.662.552l-.47 1.521c-.661-.357-1.409-.535-2.245-.535-.661 0-1.178.16-1.548.483a1.38 1.38 0 0 0-.47 1.055c0 .46.179.84.54 1.139.312.277.88.577 1.705.9 1.01.403 1.75.874 2.228 1.415.476.542.713 1.216.713 2.022m6.213-3.731h-2.071v4.076c0 1.037.365 1.555 1.096 1.555.336 0 .616-.029.836-.086l.051 1.417c-.37.137-.857.206-1.461.206-.743 0-1.323-.224-1.741-.672-.416-.45-.627-1.204-.627-2.264V20.57h-1.234v-1.4H79.9v-1.538l1.846-.553v2.09h2.071v1.4m7.398 2.79c0-.774-.168-1.438-.505-1.992-.394-.67-.958-1.005-1.687-1.005-.756 0-1.33.334-1.724 1.005-.337.554-.504 1.229-.504 2.027 0 .774.167 1.439.504 1.992.407.67.975 1.005 1.707 1.005.717 0 1.28-.34 1.687-1.022.347-.565.522-1.236.522-2.01zm1.95-.06c0 1.29-.371 2.35-1.114 3.178-.778.853-1.811 1.279-3.099 1.279-1.241 0-2.23-.409-2.966-1.227-.737-.817-1.106-1.849-1.106-3.092 0-1.301.38-2.367 1.14-3.196.76-.83 1.783-1.244 3.071-1.244 1.242 0 2.24.409 2.995 1.227.719.794 1.079 1.819 1.079 3.075zm6.108-2.489a3.281 3.281 0 0 0-.592-.05c-.661 0-1.172.247-1.532.742-.314.438-.47.99-.47 1.658v4.406h-1.878l.016-5.752c0-.969-.023-1.85-.07-2.645h1.638l.068 1.607h.052c.198-.552.512-.997.94-1.33.418-.3.871-.449 1.358-.449.174 0 .331.012.47.034v1.78m6.613 1.676c.013-.553-.11-1.03-.365-1.434-.325-.518-.825-.777-1.497-.777-.615 0-1.115.252-1.496.76-.314.403-.5.887-.556 1.45h3.914zm1.792.484c0 .334-.022.616-.069.846h-5.637c.022.83.294 1.463.817 1.901.475.391 1.09.587 1.844.587.834 0 1.596-.133 2.281-.398l.294 1.296c-.8.346-1.745.519-2.835.519-1.312 0-2.342-.384-3.09-1.15-.747-.766-1.123-1.794-1.123-3.084 0-1.267.347-2.32 1.046-3.16.73-.9 1.716-1.348 2.957-1.348 1.219 0 2.142.448 2.769 1.347.497.714.746 1.596.746 2.643zM42.263 8.783c0-.667-.178-1.18-.535-1.536-.357-.356-.877-.535-1.56-.535-.292 0-.54.02-.745.06v4.277c.114.018.322.026.624.026.707 0 1.253-.195 1.638-.585.385-.39.578-.96.578-1.707zm.974-.025c0 1.03-.31 1.805-.932 2.326-.577.48-1.394.72-2.454.72-.525 0-.975-.022-1.351-.067V6.11c.49-.079 1.02-.118 1.59-.118 1.01 0 1.77.218 2.284.654.574.492.863 1.196.863 2.113zm4.206.929c0-.38-.083-.706-.248-.978-.194-.33-.47-.495-.83-.495-.37 0-.652.165-.846.495-.165.272-.248.604-.248.995 0 .38.083.707.248.98.2.328.478.492.838.492.354 0 .63-.167.83-.501.17-.278.256-.607.256-.988zm.958-.03c0 .635-.182 1.155-.547 1.562-.383.42-.89.629-1.522.629-.61 0-1.096-.2-1.458-.603-.362-.402-.543-.909-.543-1.52 0-.639.186-1.162.56-1.57.373-.408.876-.61 1.508-.61.611 0 1.1.2 1.472.602.352.39.53.894.53 1.51zm6.925-2.027l-1.3 4.124h-.845l-.54-1.791a13.32 13.32 0 0 1-.334-1.333h-.016c-.08.453-.191.898-.334 1.333l-.573 1.79h-.855L49.307 7.63h.949l.47 1.96c.113.465.208.906.283 1.325h.016c.068-.345.182-.784.342-1.316l.59-1.969h.752l.565 1.927c.137.47.248.922.333 1.358h.026c.062-.425.157-.877.283-1.358l.504-1.927h.906m4.789 4.124h-.924V9.39c0-.728-.278-1.092-.837-1.092a.837.837 0 0 0-.667.301c-.171.2-.257.436-.257.707v2.447h-.924V8.809c0-.363-.011-.755-.033-1.18h.812l.043.645h.026c.107-.2.268-.367.478-.498.25-.155.53-.232.838-.232.387 0 .71.125.967.374.319.305.478.76.478 1.366v2.47m1.625-6.017h.923v6.017h-.923zm5.402 3.95c0-.38-.082-.706-.247-.978-.193-.33-.471-.495-.829-.495-.372 0-.654.165-.847.495-.165.272-.248.604-.248.995 0 .38.083.707.248.98.2.328.479.492.838.492.354 0 .629-.167.829-.501.172-.278.256-.607.256-.988zm.96-.03c0 .635-.183 1.155-.548 1.562-.382.42-.89.629-1.522.629-.611 0-1.097-.2-1.458-.603-.362-.402-.542-.909-.542-1.52 0-.639.186-1.162.559-1.57.373-.408.877-.61 1.508-.61.612 0 1.1.2 1.472.602.353.39.53.894.53 1.51zm3.511.679V9.7c-1.019-.017-1.528.26-1.528.832 0 .215.058.376.177.483.119.108.27.161.451.161a.927.927 0 0 0 .564-.19.777.777 0 0 0 .336-.65zm.96 1.418h-.83l-.07-.475h-.025c-.284.379-.689.569-1.214.569-.392 0-.709-.125-.949-.374a1.168 1.168 0 0 1-.325-.84c0-.504.212-.888.637-1.155.425-.265 1.023-.396 1.793-.39v-.077c0-.543-.288-.814-.863-.814-.41 0-.772.101-1.083.305l-.188-.603c.386-.237.862-.356 1.426-.356 1.086 0 1.63.569 1.63 1.707v1.519c0 .412.02.74.06.984zm4.291-1.74v-.687a1.043 1.043 0 0 0-.36-.844.908.908 0 0 0-.617-.226.952.952 0 0 0-.813.407c-.197.272-.296.62-.296 1.044 0 .408.094.739.284.993.2.271.47.408.807.408a.895.895 0 0 0 .73-.34c.178-.209.265-.46.265-.755zm.958 1.74h-.82l-.042-.662h-.027c-.262.503-.707.755-1.334.755-.501 0-.917-.195-1.248-.585-.33-.391-.496-.897-.496-1.52 0-.667.179-1.208.539-1.62a1.662 1.662 0 0 1 1.283-.578c.557 0 .948.187 1.17.56h.017V5.737h.925v4.906c0 .401.01.771.033 1.11zm6.943-2.067c0-.38-.083-.706-.248-.978-.194-.33-.47-.495-.83-.495-.37 0-.652.165-.847.495-.165.272-.248.604-.248.995 0 .38.083.707.248.98.2.328.479.492.839.492.353 0 .63-.167.83-.501.17-.278.256-.607.256-.988zm.957-.03c0 .635-.182 1.155-.547 1.562-.383.42-.889.629-1.522.629-.61 0-1.095-.2-1.458-.603-.362-.402-.542-.909-.542-1.52 0-.639.186-1.162.56-1.57.372-.408.875-.61 1.51-.61.608 0 1.1.2 1.469.602.353.39.53.894.53 1.51zm4.968 2.097h-.923V9.39c0-.728-.279-1.092-.839-1.092a.833.833 0 0 0-.666.301c-.17.2-.257.436-.257.707v2.447h-.924V8.809c0-.363-.01-.755-.033-1.18h.811l.043.645h.026c.108-.2.269-.367.478-.498.252-.155.531-.232.839-.232.388 0 .71.125.966.374.32.305.478.76.478 1.366v2.47m6.218-3.438h-1.019v2.004c0 .509.181.764.54.764.165 0 .302-.014.41-.043l.025.696c-.182.068-.422.102-.718.102-.365 0-.649-.11-.853-.331-.207-.221-.309-.591-.309-1.112v-2.08h-.608v-.687h.608v-.756l.905-.271v1.027h1.019v.687m4.89 3.438h-.924V9.409c0-.74-.28-1.11-.837-1.11-.429 0-.721.215-.882.644-.027.09-.042.2-.042.33v2.48h-.923V5.738h.923v2.485h.017c.29-.452.708-.678 1.248-.678.383 0 .699.125.95.374.314.31.47.772.47 1.383v2.453m4.163-2.497a1.23 1.23 0 0 0-.179-.704c-.16-.255-.404-.382-.735-.382a.88.88 0 0 0-.735.374 1.38 1.38 0 0 0-.274.712h1.923zm.882.238c0 .165-.013.303-.035.416h-2.77c.012.408.144.718.4.934.235.192.539.288.909.288.41 0 .783-.064 1.12-.195l.145.636c-.394.17-.857.256-1.395.256-.643 0-1.15-.188-1.516-.565-.368-.377-.55-.882-.55-1.515 0-.622.17-1.14.512-1.553.358-.442.842-.663 1.453-.663.598 0 1.052.221 1.358.663.246.35.369.784.369 1.298z" fill="#FFF"></path></g></svg></a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="community-apps.google-play" href="https://play.google.com/store/apps/details?id=com.fandom.app&amp;referrer=utm_source%3Dwikia%26utm_medium%3Dglobalfooter">
    <svg class="wds-global-footer__image wds-icon" id="wds-company-store-googleplay" viewbox="0 0 119 35" xmlns="http://www.w3.org/2000/svg"><defs><lineargradient id="store-googleplay-gradient-1" x1="91.536%" x2="-37.559%" y1="4.839%" y2="71.968%"><stop offset="0%" stop-color="#00A0FF"></stop><stop offset=".657%" stop-color="#00A1FF"></stop><stop offset="26.01%" stop-color="#00BEFF"></stop><stop offset="51.22%" stop-color="#00D2FF"></stop><stop offset="76.04%" stop-color="#00DFFF"></stop><stop offset="100%" stop-color="#00E3FF"></stop></lineargradient><lineargradient id="store-googleplay-gradient-2" x1="107.728%" x2="-130.665%" y1="49.428%" y2="49.428%"><stop offset="0%" stop-color="#FFE000"></stop><stop offset="40.87%" stop-color="#FFBD00"></stop><stop offset="77.54%" stop-color="orange"></stop><stop offset="100%" stop-color="#FF9C00"></stop></lineargradient><lineargradient id="store-googleplay-gradient-3" x1="86.389%" x2="-49.888%" y1="17.815%" y2="194.393%"><stop offset="0%" stop-color="#FF3A44"></stop><stop offset="100%" stop-color="#C31162"></stop></lineargradient><lineargradient id="store-googleplay-gradient-4" x1="-18.579%" x2="42.275%" y1="-54.527%" y2="24.69%"><stop offset="0%" stop-color="#32A071"></stop><stop offset="6.85%" stop-color="#2DA771"></stop><stop offset="47.62%" stop-color="#15CF74"></stop><stop offset="80.09%" stop-color="#06E775"></stop><stop offset="100%" stop-color="#00F076"></stop></lineargradient></defs><g fill="none" fill-rule="evenodd"><path d="M114.593 35H4.407C1.94 35 0 33.075 0 30.625V4.375C0 1.925 1.94 0 4.407 0h110.186C117.06 0 119 1.925 119 4.375v26.25c0 2.362-1.94 4.375-4.407 4.375z" fill="#000"></path><path d="M114.593.7c2.027 0 3.702 1.662 3.702 3.675v26.25c0 2.013-1.675 3.675-3.702 3.675H4.407C2.38 34.3.705 32.638.705 30.625V4.375C.705 2.362 2.38.7 4.407.7h110.186zm0-.7H4.407C1.94 0 0 1.925 0 4.375v26.25C0 33.075 1.94 35 4.407 35h110.186c2.468 0 4.407-1.925 4.407-4.375V4.375C119 2.013 117.06 0 114.593 0z" fill="#A6A6A6"></path><path d="M41.475 8.925c0 .7-.175 1.313-.613 1.75-.525.525-1.137.787-1.925.787-.787 0-1.4-.262-1.924-.787-.526-.525-.788-1.138-.788-1.925 0-.787.262-1.4.788-1.925.525-.525 1.137-.788 1.925-.788.35 0 .7.088 1.05.263.35.175.612.35.787.612l-.438.438c-.35-.438-.787-.612-1.4-.612-.524 0-1.05.174-1.4.612-.437.35-.612.875-.612 1.488 0 .612.175 1.137.613 1.487.437.35.875.613 1.4.613.612 0 1.05-.176 1.487-.613.263-.262.438-.612.438-1.05h-1.925v-.613h2.537v.263zM45.5 6.737h-2.362V8.4h2.187v.612h-2.187v1.663H45.5v.7h-3.062v-5.25H45.5zm2.888 4.638h-.7V6.737H46.2v-.612h3.675v.612h-1.487zm4.025 0v-5.25h.7v5.25zm3.674 0h-.7V6.737H53.9v-.612h3.587v.612H56v4.638zm8.313-.7c-.525.525-1.138.787-1.925.787-.788 0-1.4-.262-1.925-.787-.525-.525-.787-1.138-.787-1.925 0-.787.262-1.4.787-1.925.525-.525 1.138-.788 1.925-.788.788 0 1.4.263 1.925.788.525.525.787 1.138.787 1.925 0 .787-.262 1.4-.787 1.925zm-3.325-.438c.35.35.875.613 1.4.613.525 0 1.05-.175 1.4-.612.35-.35.612-.875.612-1.488s-.174-1.138-.612-1.487c-.35-.35-.875-.613-1.4-.613-.525 0-1.05.175-1.4.612-.35.35-.613.875-.613 1.488s.175 1.137.613 1.488zm5.075 1.138v-5.25h.788l2.537 4.113V6.125h.7v5.25h-.7l-2.712-4.287v4.287z" fill="#FFF" stroke="#FFF" stroke-width=".2"></path><path d="M59.587 19.075c-2.1 0-3.762 1.575-3.762 3.762 0 2.1 1.662 3.763 3.762 3.763s3.763-1.575 3.763-3.763c0-2.274-1.663-3.762-3.763-3.762zm0 5.95c-1.137 0-2.1-.962-2.1-2.275s.963-2.275 2.1-2.275c1.138 0 2.1.875 2.1 2.275 0 1.313-.962 2.275-2.1 2.275zm-8.137-5.95c-2.1 0-3.763 1.575-3.763 3.762 0 2.1 1.663 3.763 3.763 3.763 2.1 0 3.762-1.575 3.762-3.763 0-2.274-1.662-3.762-3.762-3.762zm0 5.95c-1.138 0-2.1-.962-2.1-2.275s.962-2.275 2.1-2.275c1.137 0 2.1.875 2.1 2.275 0 1.313-.962 2.275-2.1 2.275zm-9.713-4.813v1.576H45.5c-.087.875-.438 1.575-.875 2.012-.525.525-1.4 1.137-2.888 1.137-2.362 0-4.112-1.837-4.112-4.2 0-2.362 1.837-4.2 4.112-4.2 1.225 0 2.188.526 2.888 1.138l1.138-1.137c-.963-.876-2.188-1.575-3.938-1.575-3.15 0-5.863 2.624-5.863 5.775 0 3.15 2.713 5.774 5.863 5.774 1.75 0 2.975-.524 4.025-1.662 1.05-1.05 1.4-2.538 1.4-3.675 0-.35 0-.7-.087-.963h-5.425zm39.726 1.226c-.35-.875-1.225-2.363-3.15-2.363-1.925 0-3.5 1.488-3.5 3.762 0 2.1 1.575 3.763 3.674 3.763 1.663 0 2.713-1.05 3.063-1.663l-1.225-.875c-.438.613-.963 1.05-1.838 1.05-.874 0-1.4-.35-1.837-1.137l4.987-2.1-.174-.438zm-5.075 1.225c0-1.4 1.137-2.188 1.924-2.188.613 0 1.225.35 1.4.787l-3.325 1.4zm-4.113 3.587h1.662V15.312h-1.662V26.25zm-2.625-6.387c-.438-.438-1.138-.875-2.013-.875-1.837 0-3.587 1.662-3.587 3.762s1.663 3.675 3.588 3.675c.874 0 1.575-.438 1.924-.875h.088v.525c0 1.4-.788 2.188-2.013 2.188-.962 0-1.662-.7-1.837-1.313l-1.4.613c.438.962 1.487 2.187 3.325 2.187 1.925 0 3.5-1.137 3.5-3.85v-6.65H69.65v.613zm-1.925 5.162c-1.137 0-2.1-.962-2.1-2.275s.963-2.275 2.1-2.275c1.138 0 2.013.962 2.013 2.275s-.876 2.275-2.013 2.275zm21.35-9.712h-3.938V26.25H86.8v-4.113h2.275c1.837 0 3.587-1.312 3.587-3.412 0-2.1-1.75-3.413-3.587-3.413zm.087 5.25H86.8V16.8h2.362c1.225 0 1.925 1.05 1.925 1.837-.087.963-.787 1.925-1.925 1.925zm10.063-1.575c-1.225 0-2.45.524-2.887 1.662l1.487.613c.35-.613.875-.788 1.487-.788.876 0 1.663.525 1.75 1.4v.087c-.262-.174-.962-.437-1.662-.437-1.575 0-3.15.875-3.15 2.45 0 1.487 1.313 2.45 2.713 2.45 1.137 0 1.662-.525 2.1-1.05h.087v.875h1.575v-4.2c-.175-1.925-1.662-3.063-3.5-3.063zm-.175 6.037c-.525 0-1.313-.262-1.313-.962 0-.875.963-1.138 1.75-1.138.7 0 1.05.175 1.488.35-.175 1.05-1.05 1.75-1.925 1.75zm9.188-5.775l-1.838 4.725h-.088l-1.924-4.725h-1.75l2.887 6.65-1.663 3.675h1.663l4.462-10.325h-1.75zm-14.7 7H95.2V15.312h-1.663V26.25z" fill="#FFF"></path><path d="M9.1 6.563c-.262.262-.438.7-.438 1.224v19.338c0 .525.175.962.438 1.225l.088.087 10.85-10.85v-.174L9.1 6.563z" fill="url(#store-googleplay-gradient-1)"></path><path d="M23.625 21.262l-3.587-3.587v-.262l3.587-3.588.087.088L28 16.363c1.225.7 1.225 1.837 0 2.537l-4.375 2.363z" fill="url(#store-googleplay-gradient-2)"></path><path d="M23.712 21.175L20.038 17.5 9.1 28.438c.438.437 1.05.437 1.838.087l12.774-7.35" fill="url(#store-googleplay-gradient-3)"></path><path d="M23.712 13.825L10.938 6.563c-.788-.438-1.4-.35-1.838.087L20.038 17.5l3.674-3.675z" fill="url(#store-googleplay-gradient-4)"></path><path d="M23.625 21.087l-12.688 7.175c-.7.438-1.312.35-1.75 0l-.087.088.088.087c.437.35 1.05.438 1.75 0l12.687-7.35z" fill="#000" opacity=".2"></path><path d="M9.15 28.262c-.3-.262-.4-.7-.4-1.224v.087c0 .525.2.962.5 1.225v-.088h-.1zM28 18.637l-4.375 2.45.087.088L28 18.725c.613-.35.875-.788.875-1.225 0 .438-.35.788-.875 1.137z" fill="#000" opacity=".12"></path><path d="M10.938 6.65L28 16.363c.525.35.875.7.875 1.137 0-.438-.262-.875-.875-1.225L10.937 6.563c-1.224-.7-2.187-.088-2.187 1.312v.088c0-1.4.963-2.013 2.188-1.313z" fill="#FFF" opacity=".25"></path></g></svg></a>
    </li>
    </ul>
    </section>
    <section class="wds-global-footer__wikia-section wds-is-advertise">
    <h3 class="wds-global-footer__section-header">Advertise</h3>
    <ul class="wds-global-footer__links-list">
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="advertise.media-kit" href="//fandom.wikia.com/mediakit">
    	Media Kit</a>
    </li>
    <li class="wds-global-footer__links-list-item">
    <a class="wds-global-footer__link" data-tracking-label="advertise.contact" href="//fandom.wikia.com/mediakit#contact">
    	Contact</a>
    </li>
    </ul>
    </section>
    </div>
    </div>
    <div class="wds-global-footer__bottom-bar">
    <div class="wds-global-footer__bottom-bar-row wds-has-padding">LyricWiki is a FANDOM Music Community.</div>
    <div class="wds-global-footer__bottom-bar-row wds-has-border-top mobile-site-link">
    <a class="wds-global-footer__button-link" href="">
    		View Mobile Site	</a>
    </div>
    </div>
    </footer>
    <div id="WikiaBar">
    <div class="WikiaBarWrapper hidden" id="WikiaBarWrapper">
    <div class="wikia-bar wikia-bar-anon">
    <a class="arrow" data-tooltip="Collapse" data-tooltipshow="Show" href="#"></a>
    <div class="ad">
    <div class="noprint" id="WIKIA_BAR_BOXAD_1" style="width: 300px; position: relative;"></div>
    </div>
    <div class="message" data-messagetooltip="Click here for more information!" data-wikiabarcontent='[{"text":"These are the RDR2 side quests you need to play!","href":"http:\/\/bit.ly\/2TYzFDU"},{"text":"These are the RDR2 side quests you need to play!","href":"http:\/\/bit.ly\/2TYzFDU"},{"text":"These are the RDR2 side quests you need to play!","href":"http:\/\/bit.ly\/2TYzFDU"},{"text":"These are the RDR2 side quests you need to play!","href":"http:\/\/bit.ly\/2TYzFDU"},{"text":"These are the RDR2 side quests you need to play!","href":"http:\/\/bit.ly\/2TYzFDU"}]'></div>
    <a class="wikiabar-button" data-index="0" href="http://bit.ly/2FJkLho">
    <img class="icon cup" src="https://slot1-images.wikia.nocookie.net/__cb7800027800012/common/extensions/wikia/WikiaBar/images/wikiabarIcon.png">
    <span>PC Gift Guide</span>
    </img></a>
    <a class="wikiabar-button" data-index="1" href="http://bit.ly/2S4r0xW">
    <img class="icon cup" src="https://slot1-images.wikia.nocookie.net/__cb7800027800012/common/extensions/wikia/WikiaBar/images/wikiabarIcon.png">
    <span>Game Streams</span>
    </img></a>
    <a class="wikiabar-button" data-index="2" href="http://bit.ly/2DQtTz3">
    <img class="icon cup" src="https://slot1-images.wikia.nocookie.net/__cb7800027800012/common/extensions/wikia/WikiaBar/images/wikiabarIcon.png">
    <span>Fallout 76</span>
    </img></a>
    </div>
    </div>
    <div class="WikiaBarCollapseWrapper">
    <a class="wikia-bar-collapse" data-tooltip="Collapse" href="#"></a>
    </div>
    </div>
    <!-- BEGIN SLOTNAME: GPT_FLUSH -->
    <div class="wikia-ad noprint default-height" id="GPT_FLUSH">
    <script>
    							window.adslots2.push(["GPT_FLUSH"]);
    					</script>
    </div>
    <!-- END SLOTNAME: GPT_FLUSH -->
    </div>
    <!--[if lt IE 8]>
    		<script src="https://slot1-images.wikia.nocookie.net/__cb7800027800012/common/resources/wikia/libraries/json2/json2.js"></script>
    	<![endif]-->
    <!--[if lt IE 9]>
    		<script src="https://slot1-images.wikia.nocookie.net/__cb7800027800012/common/resources/wikia/libraries/html5/html5.min.js"></script>
    	<![endif]-->
    <!-- Combined JS files and head scripts -->
    <script src="/load.php?cb=7800027800012&amp;debug=false&amp;lang=en&amp;modules=startup&amp;newve=1&amp;only=scripts&amp;skin=oasis&amp;*"></script>
    <script>if(window.mw){
    mw.config.set({"wgCanonicalNamespace":"","wgCanonicalSpecialPageName":false,"wgPageName":"Missy_Elliott:Lose_Control","wgTitle":"Missy Elliott:Lose Control","wgCurRevisionId":30818251,"wgArticleId":275575,"wgIsArticle":true,"wgAction":"view","wgUserName":null,"wgUserGroups":["*"],"wgCategories":["Song","Green Songs","Language/English","Songs by Missy Elliott","Songs L","ITunes/Song","Spotify/Song","Allmusic/Song","MusicBrainz/Song"],"wgBreakFrames":false,"wgPageContentLanguage":"en","wgSeparatorTransformTable":["",""],"wgDigitTransformTable":["",""],"wgRelevantPageName":"Missy_Elliott:Lose_Control","wgRestrictionEdit":[],"wgRestrictionMove":[],"sassParams":{"background-dynamic":"false","background-image":"https://vignette3.wikia.nocookie.net/lyricwiki/images/5/50/Wiki-background/revision/latest?cb=20140425192856","background-image-height":"1080","background-image-width":"2100","color-body":"#380759","color-body-middle":"#fff","color-buttons":"#006cb0","color-community-header":"#380759","color-header":"#3a5766","color-links":"#006cb0","color-page":"#ffffff","oasisTypography":1,"page-opacity":"100","widthType":0},"wgAssetsManagerQuery":"/__am/%4$d/%1$s/%3$s/%2$s","wgWeppyConfig":{"host":"https://speed.nocookie.net/__rum","sample":0.1,"aggregationInterval":1000},"WikiaEnableNewCreatepage":true,"ContentNamespacesText":["",false],"wgCatId":"16","wgBlankImgUrl":"data:image/gif;base64,R0lGODlhAQABAIABAAAAAP///yH5BAEAAAEALAAAAAABAAEAQAICTAEAOw%3D%3D","wgPrivateTracker":true,"wgMainpage":"LyricWiki","wgIsContentNamespace":true,"wgExtensionsPath":"https://slot1-images.wikia.nocookie.net/__cb7800027800012/common/extensions","wgResourceBasePath":"https://slot1-images.wikia.nocookie.net/__cb7800027800012/common","wgSitename":"LyricWiki","wgMWrevId":false,"wgRevisionId":30818251,"wgEnableNewAuthModal":true,"wgEnableWikiaPhotoGalleryExt":true,"wgOasisGrid":true,"wgEnableMediaGalleryExt":false,"wgWikiaMaxNameChars":50,"wgMinimalPasswordLength":1,"wgEnableLightboxExt":true,"wgEnableWikiaFollowedPages":true,"wgFollowedPagesPagerLimit":15,"wgFollowedPagesPagerLimitAjax":600,"wgWikiaChatUsers":[],"wgWikiaChatWindowFeatures":"width=600,height=600,menubar=no,status=no,location=no,toolbar=no,scrollbars=no,resizable=yes","wgTrackID":0,"wgEnableWikiaBarExt":true,"wgEnableWikiaBarAds":true,"wgWikiaBarMainLanguages":["de","en","es","fr"],"wgVisualEditor":{"isPageWatched":false,"pageLanguageCode":"en","pageLanguageDir":"ltr","svgMaxSize":2048,"namespacesWithSubpages":{"0":false,"1":true,"2":true,"3":true,"4":true,"5":true,"7":true,"6":true,"8":true,"9":true,"10":true,"11":true,"12":true,"13":true,"14":true,"15":true,"222":false,"110":true,"111":true,"828":true,"829":true,"1201":true,"2001":true,"500":true,"501":true,"502":true,"503":true}},"wgMaxUploadSize":10485760,"wgEnableVisualEditorUI":false,"wgEnableWikiaInteractiveMaps":false,"VignettePathPrefix":null,"reCaptchaPublicKey":"6LdDSA4TAAAAANZDWjPdTiQcYsTuge5fMPQTd7D_","wgQualarooUrl":"//s3.amazonaws.com/ki.js/52510/bgJ.js","isContributor":false,"isCurrentWikiAdmin":false,"fullVerticalName":"music","dartGnreValues":[],"wgVisualEditorPreferred":false,"wgEnablePortableInfoboxBuilderInVE":false,"egMapsDebugJS":false,"egMapsAvailableServices":["googlemaps3","openlayers","leaflet"],"wgOasisResponsive":false,"wgOasisBreakpoints":true,"verticalName":"Entertainment","wgArticleInterlangList":[],"wgCategoryTreePageCategoryOptions":"{\"mode\":0,\"hideprefix\":20,\"showcount\":true,\"namespaces\":false}","wgContentReviewExtEnabled":true,"wgContentReviewTestModeEnabled":false,"wgReviewedScriptsTimestamp":"1544041404","wgScriptsTimestamp":"1544016041","wgCategorySelect":{"defaultNamespace":"Category","defaultNamespaces":"Category","defaultSeparator":":","defaultSortKey":"Missy Elliott:Lose Control"}});
    }</script><script>if(window.mw){
    mw.loader.implement("user.options",function($){mw.user.options.set({"ccmeonemails":0,"cols":80,"date":"default","diffonly":0,"disablemail":0,"disablesuggest":0,"editfont":"default","editondblclick":0,"editor":2,"editsection":1,"editsectiononrightclick":0,"enotifdiscussionsfollows":1,"enotifdiscussionsvotes":1,"enotifminoredits":1,"enotifrevealaddr":0,"enotifusertalkpages":1,"enotifwatchlistpages":1,"extendwatchlist":0,"externaldiff":0,"externaleditor":0,"forceeditsummary":0,"hideminor":0,"hidepatrolled":0,"highlightbroken":1,"htmlemails":1,"imagesize":1,"justify":0,"math":6,"minordefault":0,"newpageshidepatrolled":0,"nocache":0,"noconvertlink":0,"norollbackdiff":0,"numberheadings":0,"previewonfirst":0,"previewontop":1,"quickbar":5,"rcdays":7,"rclimit":50,"rememberpassword":0,"rows":25,"searchlimit":20,"showhiddencats":0,"showjumplinks":1,"shownumberswatching":1,"showtoc":1,"showtoolbar":1,"skin":"oasis","stubthreshold":0,"thumbsize":2,"underline":2,"uselivepreview":0,"usenewrc":1,
    "watchcreations":1,"watchdefault":1,"watchdeletion":1,"watchlistdays":3,"watchlistdigest":1,"watchlisthideanons":0,"watchlisthidebots":0,"watchlisthideliu":0,"watchlisthideminor":0,"watchlisthideown":0,"watchlisthidepatrolled":0,"watchmoves":0,"wllimit":250,"visualeditor-enable":1,"visualeditor-betatempdisable":0,"variant":"en","language":"en","searchNs0":!0,"searchNs1":!1,"searchNs2":!1,"searchNs3":!1,"searchNs4":!1,"searchNs5":!1,"searchNs6":!1,"searchNs7":!1,"searchNs8":!1,"searchNs9":!1,"searchNs10":!1,"searchNs11":!1,"searchNs12":!1,"searchNs13":!1,"searchNs14":!1,"searchNs15":!1,"searchNs110":!1,"searchNs111":!1,"searchNs112":!1,"searchNs113":!1,"searchNs222":!1,"searchNs223":!1,"searchNs500":!1,"searchNs501":!1,"searchNs502":!1,"searchNs503":!1,"searchNs828":!1,"searchNs829":!1,"searchNs1200":!1,"searchNs1201":!1,"searchNs1202":!1,"userlandingpage":1,"category-page-layout":
    "category-page3"});;},{},{});mw.loader.implement("user.tokens",function($){mw.user.tokens.set({"editToken":"+\\","watchToken":!1});;},{},{});
    
    /* cache key: lyricwiki:resourceloader:filter:minify-js:7:64fd4c73499fe823055af2bfc4ab9c61 */
    }</script>
    <script>if(window.mw){
    mw.loader.load(["mediawiki.page.startup","mediawiki.legacy.wikibits","mediawiki.legacy.ajax","amd.shared","ext.visualEditor.wikia.viewPageTarget.init","mediawiki.user","mediawiki.page.ready","ext.wikia.TimeAgoMessaging","ext.designSystem","ext.bannerNotifications","ext.wikia.facebookTags","ext.userLogin"]);
    }</script><script src="https://slot1-images.wikia.nocookie.net/__am/7800027800012/groups/-/oasis_shared_core_js,oasis_shared_js,oasis_anon_js,toc_js,LyricsFindTracking,adengine2_desktop_js,recirculation_liftigniter_tracker,recirculation_js,qualaroo_js"></script>
    <script src="https://slot1-images.wikia.nocookie.net/__am/7800027800012/group/-/auth_modal_js"></script>
    <script src="https://slot1-images.wikia.nocookie.net/__am/7800027800012/group/-/community_header_js"></script>
    <script src="https://slot1-images.wikia.nocookie.net/__am/7800027800012/group/-/page_header_js"></script>
    <script src="https://slot1-images.wikia.nocookie.net/__am/7800027800012/group/-/wikia_in_your_lang_js"></script>
    <script src="https://slot1-images.wikia.nocookie.net/__am/7800027800012/group/-/portable_infobox_js"></script>
    <script src="https://slot1-images.wikia.nocookie.net/__am/7800027800012/group/-/visit_source_js"></script>
    <script src="/wikia.php?controller=JSMessages&amp;method=getMessages&amp;format=html&amp;packages=AdEngine%2CArticleVideo%2CConfirmModal%2CImagePlaceholder%2COasis-generic%2CRecirculation&amp;uselang=en&amp;cb=7800027800012.32311183.0"></script>
    <script id="liftigniter-metadata" type="application/json">{"language":"en"}</script><script type="text/javascript">/*<![CDATA[*/ Wikia.LazyQueue.makeQueue(wgAfterContentAndJS, function(fn) {fn();}); wgAfterContentAndJS.start(); /*]]>*/</script>
    <script type="text/javascript">/*<![CDATA[*/ if (typeof AdEngine_trackPageInteractive === 'function') {wgAfterContentAndJS.push(AdEngine_trackPageInteractive);} /*]]>*/</script>
    <script src="/load.php?cb=7800027800012&amp;debug=false&amp;lang=en&amp;modules=ext.siteWideMessages.anon&amp;only=scripts&amp;skin=oasis&amp;*"></script>
    <script src="/load.php?cb=7800027800012&amp;debug=false&amp;lang=en&amp;modules=site&amp;only=scripts&amp;reviewed=1544041404&amp;skin=oasis&amp;*"></script>
    <script defer="" src="https://www.fastly-insights.com/static/scout.js?k=17272cd8-82ee-4eb5-b5a3-b3cd5403f7c5"></script><script>(function(){importWikiaScriptPages(["external:dev:CodeLoad.js"]);})();</script><script>var wgSassLoadedScss = ["skins\/oasis\/css\/oasis.scss","extensions\/wikia\/DesignSystem\/styles\/design-system.scss","extensions\/wikia\/CommunityHeader\/styles\/index.scss","extensions\/wikia\/PageHeader\/styles\/index.scss","extensions\/wikia\/Recirculation\/styles\/recirculation.scss","extensions\/wikia\/PortableInfobox\/styles\/PortableInfobox.scss","extensions\/wikia\/PortableInfobox\/styles\/PortableInfoboxEuropaTheme.scss","extensions\/wikia\/Qualaroo\/css\/Qualaroo.scss"];</script>
    </body>
    </html>



## 2. Data Resampling

### (1) 

This is converted encoded as either 0 or 1 for each of the 126 Spotify seed genres. Seed genres can be obtained by Spotipy using spotipy.Spotify().recommendation_genre_seeds().
    - More information is available at: https://developer.spotify.com/documentation/web-api/reference/artists/get-artist/
