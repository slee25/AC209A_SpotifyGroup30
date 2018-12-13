---
title: Literature Review
notebook: 
nav_include: 4
---


## Contents
{:.no_toc}
*  
{: toc}


## List of Reference

- We have reviewed 30+ papers that are important in the music recommendation field, including:
	- Logan, Beth, and Ariel Salomon. "A content-based music similarity function." Cambridge Research Labs-Tech Report (2001).
	- Knees, Peter, and Markus Schedl. "A survey of music similarity and recommendation from music context data." ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 10.1 (2013): 2.
	- Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. Ieee, 2008.
	- Casey, Michael A., et al. "Content-based music information retrieval: Current directions and future challenges." Proceedings of the IEEE 96.4 (2008): 668-696.
	- Schedl, Markus, and Peter Knees. "Context-based music similarity estimation." Welcome to the 3 rd International Workshop on Learning Semantics of Audio Signals. 2009.
	- Schedl, Markus, et al. "Current challenges and visions in music recommender systems research." International Journal of Multimedia Information Retrieval 7.2 (2018): 95-116.
	- Van Der Maaten, Laurens, Eric Postma, and Jaap Van den Herik. "Dimensionality reduction: a comparative." J Mach Learn Res 10 (2009): 66-71.
	- Jannach, Dietmar, Lukas Lerche, and Iman Kamehkhosh. "Beyond hitting the hits: Generating coherent music playlist continuations with the right tracks." Proceedings of the 9th ACM Conference on Recommender Systems. ACM, 2015.
	- Berenzweig, Adam, et al. "A large-scale evaluation of acoustic and subjective music-similarity measures." Computer Music Journal 28.2 (2004): 63-76.
	- Yang, Hojin, et al. "MMCF: Multimodal Collaborative Filtering for Automatic Playlist Continuation." Proceedings of the ACM Recommender Systems Challenge 2018. ACM, 2018.
	- Schedl, Markus, Peter Knees, and Fabien Gouyon. "New paths in music recommender systems research." Proceedings of the Eleventh ACM Conference on Recommender Systems. ACM, 2017.
	- Vagliano, Iacopo, et al. "Using Adversarial Autoencoders for Multi-Modal Automatic Playlist Continuation." Proceedings of the ACM Recommender Systems Challenge 2018. ACM, 2018.
	
	
- Below, we summarize a couple of papers which we believe are the most relevant to our projects.
 
## MMCF: Multimodal Collaborative Filtering for Automatic Playlist Continuation

- This paper proposes a multimodal collaborative filtering model to deal effectively with diverse data. This consists of two components: (1) an autoencoder using both the playlist and its categorical contents and (2) a character-level convolutional neural network using the playlist title only. The unique feature of this paper is that it discussed the use of autoencoder to extract the content of the playlists. Though the actual methods are too complicated to implement, this gives us a hint to use autoencoder and modify it to train the model other than simply matching the original data, which is the original purpose of autoencoder.

## Content-Based Music Similarity Function

- This paper presents a method to compare songs based solely on their audio content. Their technique forms a signature for each song based on K-means clustering of spectral features. The signatures can then be compared using the Earth Mover’s Distance which allows comparison of histograms with disparate bins. The paper analyzed the acoustic data that we do not implement, while it uses MDS to visualize the relationship between the tracks.

## A Large-Scale Evaluation of Acoustic and Subjective Music-Similarity Measures

- The paper introduced the similarity matrices and includes different metadata that is beyond the acoustic data itself. They found that: (1) Acoustic-based measures can achieve agreement with ground truth data that is at least comparable to the internal agreement between different subjective sources. (2) Subjective measures from diverse sources show reasonable agreement, with the measure derived from co-occurrence in personal music collections being the most reliable overall. (3) Their methodology for largescale cross-site music similarity evaluations is practical and convenient, yielding directly comparable numbers for different approaches.

## Beyond “Hitting the Hits” – Generating Coherent Music Playlist Continuations with the Right Tracks

- The paper discussed the "score" from different features that represent the relationship between the playlist and the candidate tracks. They propose an algorithmic approach and optimization scheme to generate playlist continuations that address these requirements. In their approach, they first use collections of shared music playlists, music metadata, and user preferences to select suitable tracks with high accuracy. Next, they apply a generic re-ranking optimization scheme to generate playlist continuations that match the characteristics of the last played tracks.