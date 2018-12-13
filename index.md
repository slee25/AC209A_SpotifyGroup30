---
title: AC209A Final Project - Automatic Playlist Generation
---

by Chih-Kang Chang, Jinsoo Kim, and Sangjun Lee (Group #30)

## Contents
{:.no_toc}
*  
{: toc}

## Motivation

Music recommender systems (MRS) have recently exploded in popularity thanks to music streaming services like Spotify, Pandora and Apple Music. By some accounts, almost half of all current music consumption is by the way of these services. While recommender systems have been around for quite some time and are very well researched, music recommender systems differ from their more common siblings in some characteristically important ways: the duration of the items is less (3-5 min for a song vs 90 minutes for a movie or months/years for a book or shopping item), the size of the catalog of items is larger (10s of millions of songs), the items are consumed in sequence with multiple items consumed in a session, repeated recommendations have a different significance since listening to the same song as part of different playlists may be ok, and consumption occurs passively i.e. in the background. Music Recommender Systems then require different approaches from traditional recommender systems.

<img src="https://cdn-images-1.medium.com/max/800/1*HhgUcC9pvO592FHG_s91zA.jpeg" width="600" height="250" align="middle">

One of the major problems in Music Recommender Systems is the station/playlist generation problem. At its heart, the playlist generation is about finding the set of songs to recommend to best extend the experience of a listener in the midst of a playlist. By suggesting appropriate songs to add to a playlist, a Recommender System can increase user engagement by making playlist creation easier, as well as extending listening beyond the end of existing playlists.

One of Spotify’s primary products is Playlists, collections of tracks that individual users (or Spotify) can build for every mood or event. Spotify users can make or follow as many playlists as they like. With over 40 million songs available, the company attempts to direct the most relevant songs to users based on their preferences, and Playlists often comprise the most convenient and effective way to convey these recommended songs to a user.

## Problem Statement

The goal of this project is to develop a system for the task of automatic playlist continuation. Given a set of playlists (“targeted playlist”), our algorithm shall generate a list of recommended tracks that can be added, thereby ‘continuing’ the playlist. For the training data, we would have the playlists metadata as the inputs. Based on a list of the K tracks in the targeted playlist with N tracks, where K can equal to 20%, 40%, 60%, or 80% of N, the model should compare the targeted playlist and other tracks, and generate a list of (N-K)*20 recommended tracks, ordered by relevance in decreasing order, that tries to match the remaining (N-K) tracks in the targeted playlist.

For the test data, we would have a list of K tracks in the targeted playlist, where K can equal 0, 1, 5, 10, or 25, and generate a list of (N-K)*20 recommended tracks, ordered by relevance in decreasing order, that tries to match the remaining (N-K) tracks in the targeted playlist. In this project, we have decided to use K = 25, N = 55.