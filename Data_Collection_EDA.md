---
title: Data Collection and EDA
notebook: notebooks/Data_Collection_EDA.ipynb
nav_include: 1
---


## Contents
{:.no_toc}
*  
{: toc}




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



```python
ARTIST_URI = playlist_data['playlists'][0]['tracks'][0]['artist_uri']
sp.artist(ARTIST_URI)
```


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



```python
TRACK_URI = playlist_data['playlists'][0]['tracks'][0]['track_uri']
sp.audio_features(TRACK_URI)
```


- `spotipy.Spotify().artist_related_artists(artist_uri)`: Get Spotify catalog information about artists similar to a given artist. Similarity is based on analysis of the Spotify community's listening history.
(https://developer.spotify.com/documentation/web-api/reference/artists/get-related-artists/)
    - `artists` (array of `artist objects`): Up to 20 related artists to the artist.


- Below shows an example data provided by `spotipy.Spotify().artist_related_artists(artist_uri)` for the artist of the 0th track in the 0th playlist in the 0th file:


```python
ARTIST_URI = playlist_data['playlists'][0]['tracks'][0]['artist_uri']
sp.artist_related_artists(ARTIST_URI)
```





- `spotipy.Spotify().artist_top_tracks(artist_uri)`: Get Spotify catalog information about an artist’s top tracks by country.
(https://developer.spotify.com/documentation/web-api/reference/artists/get-artists-top-tracks)
    - `tracks` (array of `track objects`): Up to 10 top tracks of the artist.


- Below shows an example data provided by `spotipy.Spotify().artist_top_tracks(artist_uri)` for the artist of the 0th track in the 0th playlist in the 0th file:


```python
ARTIST_URI = playlist_data['playlists'][0]['tracks'][0]['artist_uri']
sp.artist_top_tracks(ARTIST_URI)
```





- Notes:
  - There may be high multicollinearity in these data, as these are a mix of primary and secondary information. For example, Spotify Web API explains that the danceability is based on other primary data including tempo, while the energy is based on some other primary data including loudness. Therefore, it would be important to find a method that are insensitive to the multicollinearity among columns.

### (3) Genius API

- Genius (https://genius.com/) is an American digital media library, particularly specialized for song lyrics and musical knowledge. Genius provides public Genius API where we can access lyrics data for songs.


- Below shows a screen capture of https://genius.com/Missy-elliott-lose-control-lyrics, which is the 0th track in the 0th playlist in the 0th file, along with the developer tool.


<img src="https://raw.githubusercontent.com/slee25/AC209A_SpotifyGroup30/master/notebooks/fig/Genius.png" width="635" height="460">


- Below shows a html-parsed result of this page by using `requests` and `BeautifulSoup`:


```python
GeniusURL = "https://genius.com/Missy-elliott-lose-control-lyrics"
BeautifulSoup(requests.get(GeniusURL).text,'html.parser')
```





- Using Genius API, this information is more easily accessible.


```python
requests.get('https://api.genius.com', data={'q': 'Lose Control Missy Elliott'}, headers={'Authorization': 'Bearer ' + Genius_TOKEN}).text
```



```python
requests.get('https://api.genius.com', data={'q': 'Lose Control Missy Elliott'}, headers={'Authorization': 'Bearer ' + Genius_TOKEN}).text
```


### (4) Lyrics Wiki

- Lyrics Wiki (http://lyrics.wikia.com/wiki/LyricWiki) is a large database for lyrics. Lyrics for a certain song can be scraped at http://lyrics.wikia.com/wiki/[artist_name]:[track_name].


- Below shows a screen capture of http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control, which is the 0th track in the 0th playlist in the 0th file, along with the developer tool.
<img src="https://raw.githubusercontent.com/slee25/AC209A_SpotifyGroup30/master/notebooks/fig/LyricsWiki.png" width="635" height="460">


- Below shows a html-parsed result of this page by using `requests` and `BeautifulSoup`:


```python
LyricsWikiURL = "http://lyrics.wikia.com/wiki/Missy_Elliott:Lose_Control"
BeautifulSoup(requests.get(LyricsWikiURL).text,'html.parser')
```





## 2. Data Resampling

- As mentioned earlier, the Million Playlist Dataset is too large to handle, we have decided to resize/resample the data.

- We have decided to sample 1000 playlists among the 1 million playlists, and the selection criteria were as follows:
    - The number of followers should be equal or greater than 20.
        - If the number of followers is too few, e.g. 1 only the person who created the playlist, the taste of the playlist might be highly biased and/or inconsistent. To ensure the consistency and generalizability, we set a lower bound to the number of followers.
    - The number of tracks should be greater than 55.
        - If there are too few number of tracks in a playlist, it would be very hard to split the train and test sets, preventing meaningful predictions. We set the minimum size of both training and test sets as 25, with margin 5, so the minimum number of tracks was determined as 55.
    - Among the playlists that meets the criteria above, randomly sample 1000 playlists.
        - We may select 1000 playlists in the order of the number of followers, but in this way, the samples may also be biased by having too popular musics only.

- The `playlist_stat.npy` below contains pre-prepared data (separate ipynb and pdf are attached) for the number of followers and the number of tracks in the Million Playlist:


```python
playlist_stat = np.load('data/playlist_stat.npy')
```


- The distribution of the number of tracks in a playlist for the Million Playlist Dataset is like below:


```python
plt.figure(figsize=[8,6])
plt.hist(playlist_stat[:,4],bins=50)
plt.title('distribution of the number of tracks in a playlist')
plt.xlabel('counts of playlist')
plt.ylabel('number of tracks')
plt.show()

from statistics import mode
mode(playlist_stat[:,4])
```



```python
plt.figure(figsize=[8,6])
plt.hist(playlist_stat[:,4],bins=50)
plt.title('distribution of the number of tracks in a playlist')
plt.xlabel('counts of playlist')
plt.ylabel('number of tracks')
plt.show()

from statistics import mode
mode(playlist_stat[:,4])
```


- Thus, the number of tracks in a playlist shows a right-skewed distribution and the most fequent number is 20.

- Similarly, the distribution of the number of followers of a playlists for the Million Playlist Dataset is like below:


```python
plt.figure(figsize=[8,6])
plt.hist(playlist_stat[:,3], bins=50, log=True)
plt.title('distribution of the number of followers in a playlist (log-scale)')
plt.xlabel('counts of playlist')
plt.ylabel('number of followers')
plt.show()
```



```python
plt.figure(figsize=[8,6])
plt.hist(playlist_stat[:,3], bins=50, log=True)
plt.title('distribution of the number of followers in a playlist (log-scale)')
plt.xlabel('counts of playlist')
plt.ylabel('number of followers')
plt.show()
```


- Note that, this is in log scale. As shown, a majority of the playlists has 1 follower (the creater of the playlist themselves), and only few playlists has 10000+ followers.

- Based on the criteria above, we've sampled playlists that have 55+ tracks, and eligible playlists are 447703 playlists out of 1 million.


```python
playlist_stat_55 = playlist_stat[playlist_stat[:,4]>55,:]
playlist_stat_55.shape[0]
```



```python
playlist_stat_55 = playlist_stat[playlist_stat[:,4]>55,:]
playlist_stat_55.shape[0]
```


- Among the 447703 playlists selected, depending on the number of followers that we select as the lower bound, the final number of eligible playlists is like below:


```python
print('[followers]\t[playlists]')
print(40, '\t\t', sum(playlist_stat_55[:,3]>=40))
print(30, '\t\t', sum(playlist_stat_55[:,3]>=30))
print(20, '\t\t', sum(playlist_stat_55[:,3]>=20))
print(10, '\t\t', sum(playlist_stat_55[:,3]>=10))
print(5, '\t\t', sum(playlist_stat_55[:,3]>=5))
```



```python
print('[followers]\t[playlists]')
print(40, '\t\t', sum(playlist_stat_55[:,3]>=40))
print(30, '\t\t', sum(playlist_stat_55[:,3]>=30))
print(20, '\t\t', sum(playlist_stat_55[:,3]>=20))
print(10, '\t\t', sum(playlist_stat_55[:,3]>=10))
print(5, '\t\t', sum(playlist_stat_55[:,3]>=5))
```


- This indicates that, if we select the lower bound of the number of followers as 20 based on the criteria above, 1852 playlists out of 1 million will be eligible. We believe 1852 to 1000 is a reasonable down-sampling rate, so we sampled 1000 playlists among these 1852 playlists by using `np.random.seed(0)` and `np.random.choice` (separate ipynb and pdf are attached).

## 3. Scaping and Constructing Data Structure

### (1) Initial Data Structure from Million Playlist Dataset

- After sampling 1000 playlists out of eligible 1852 playlists, we constructed initial data structure as a Pandas DataFrame. The initial data structure contains all the information provided by the Million Playlist Dataset.
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
- For each playlist, we select the first 55 tracks, so that all playlists have the same contribution to the model. This will makes the number of tracks 550000.

- The `playlists_df_sort.pkl` below contains pre-prepared data (separate ipynb and pdf are attached) for initial data structure contains 550000 tracks from the 1000 playlists.


```python
playlists_df_sort = pd.read_pickle('data/playlists_df_sort.pkl')
display(playlists_df_sort.head())
display(playlists_df_sort.describe())
display(playlists_df_sort.shape)
```



```python
playlists_df_sort = pd.read_pickle('data/playlists_df_sort.pkl')
display(playlists_df_sort.head())
display(playlists_df_sort.describe())
display(playlists_df_sort.shape)
```


- As shown, the initial data structure contains 55000 track data (1000 playlists * 55 tracks/playlist), and the number of columns is 19, but only 10 of them are numeric, indicating there are not enough numeric data to build model and make prediction.

### (2) Data Scraping from Spotify API

- As mentioned above, by using `album_uri`, `artist_uri`, and `track_uri` in the intial data structure, we can scrape more data using Spotify API. We have created custom python functions to scrape information using different Spotify API function endpoints.

- `AddAudioFeatures`: adds `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, and `time_signature` of the track by using `spotipy.Spotify().audio_features(track_uri)`:


```python
def AddAudioFeatures(df, ind, sp):
    try:
        audio_feature = sp.audio_features(df.loc[ind]['track_uri'])[0]
    except:
        print('No ID at ' + str(ind))
    else:
        feature_list = ['danceability','energy','key','loudness','mode','speechiness','acousticness',
                        'instrumentalness','liveness','valence','tempo','time_signature']

        for feature in feature_list:
            df.at[ind, feature] = audio_feature[feature]
```


- `AddTrackPopularity`: adds `popularity` of the track by using `spotipy.Spotify().track(track_uri)`:


```python
def AddTrackPopularity(df, ind, sp):
    try:
        track_info = sp.track(df.loc[ind]['track_uri'])
    except:
        print('No ID at ' + str(ind))
    else:
        df.at[ind, 'popularity'] = track_info['popularity']
```


- `AddArtistGenres`: adds genres of the artist with one-hot encoding using `spotipy.Spotify().artist(artist_uri)`. Note that this is genres of the artist, not of the track, as the Spotify API doesn't provide genre for each track. Also, note that as a list of genres, we used `spotipy.Spotify().recommendation_genre_seeds()['genres']`, which contains names of 126 basic genres.


```python
def AddArtistGenres(df, ind, sp):
    try:
        artist_info = sp.artist(df.loc[ind]['artist_uri'])
    except:
        print('No ID at ' + str(ind))
    else:
        genre_list = sp.recommendation_genre_seeds()['genres']

        for gen in genre_list:
            if gen in artist_info['genres']:
                df.at[ind, gen] = 1
            else:
                df.at[ind, gen] = 0
```


- `AddAlbumYear`: adds the year when the album came out using `spotipy.Spotify().album(album_uri)`. Note that Spotify API provides the `release_date` in `YYYY-MM-DD` form, so the first part of the string is converted into an integer.


```python
def AddAlbumYear(df, ind, sp):
    try:
        album_info = sp.album(df.loc[ind]['album_uri'])
    except:
        print('No ID at ' + str(ind))
    else:
        df.at[ind, 'year'] = int(album_info['release_date'].split('-')[0])
```


- `AddTopTracks`: adds the number of top tracks of the artist as well as URIs of all of the top tracks using `spotipy.Spotify().artist_top_tracks(artist_uri)`. If the number of top tracks is less than 10, the remainings will be filled with blank.


```python
def AddTopTracks(df, ind, sp):    
    try:
        top_track_info = sp.artist_top_tracks(df.loc[ind]['artist_uri'])['tracks']
    except:
        print('\n No ID at ' + str(ind) + '\n')
        print(df.loc[ind]['artist_uri'])
    else:
        n = len(top_track_info)
        df.at[ind, 'top_track_n'] = n
        
        for i in range(10):
            if i in range(n):
                df.at[ind, 'top_track_'+str(i)] = top_track_info[i]['uri']
            else:
                df.at[ind, 'top_track_'+str(i)] = ''
```


- AddRelatedArtists: adds the number of related artists of the artist as well as URIs of all of the related artists using `spotipy.Spotify().artist_related_artists(artist_uri)`. If the number of related artists is less than 20, the remainings will be filled with blank.


```python
def AddRelatedArtists(df, ind, sp):
    try:
        related_artist_info = sp.artist_related_artists(df.loc[ind]['artist_uri'])['artists']
    except:
        print('\n No ID at ' + str(ind) + '\n')
        print(df.loc[ind]['artist_uri'])
    else:
        n = len(related_artist_info)
        df.at[ind, 'related_artist_n'] = n
        
        for i in range(20):
            if i in range(n):
                df.at[ind, 'related_artist_'+str(i)] = related_artist_info[i]['uri']
            else:
                df.at[ind, 'related_artist_'+str(i)] = ''
```


### (3) Data Scraping from Genius API and Lyrics Wiki

- As mentioned above, by us


```python

```

