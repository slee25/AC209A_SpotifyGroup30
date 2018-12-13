---
title: Summary and Conclusion
notebook: 
nav_include: 6
---


## Contents
{:.no_toc}
*  
{: toc}


## Summary and Conclusion

- The goal of the project is to perform the task of automatic playlist continuation. We refer to the database from the Spotify RecSys Challenge. Using techniques we learned in the class to scrape information from websites, use Spotify API to gather more information about the tracks, and clean up the data. We also investigate Natural Language Processing to analyze lyrics and expand our feature space. 

- According to past works in the field and the inspiration from the course, we establish our model based on the autoencoder. The autoencoder successfully from the features extracts the information related to the playlist continuation. The performance of our model outperforms the basic model that directly uses the raw feature space. 

- There are more works could be done to improve the results. We can extract more features from the known information that may be useful to strengthen the connection between tracks in the same playlist. There are also many hyper-parameters that can be tuned to improve our model, that we did not fully explore due to the limitation of time and effort. In addition, there exist other models that can perform the recommendation tasks that worth comparison.

- In conclusion, our model achieves the goal to automatically continue the playlist, while there are more to be explored.  

## Discussion and Future Work

### 1) Feature engineering - ‘related artist’

- In this project, we couldn't find a good way to incorporate 'related artists' (of the artist) information in a DataFrame structure. This information is definitely useful to find a connection between the tracks, by checking artists of them. The problem is that this is categorical features with too many categories (=artists). Thus, if we just incorporate this using similar concept of 'one-hot encoding', then it will adds a ton of features. 

- Possibly, “connecting graph” approach could be a solution for this problem. The nodes of connecting graph are artists, and two nodes are connected only if one artist is listed as 'related artist' of the other artist. Then, dense connections mean that all these authors in these dense clusters are closely related, so we can assign a single category for all these authors. By splitting the graph at the sparse connection, we could reduce the number of categories as a reasonable manner. Instead of directly using one hot encoding, we can reduce the number of categories first and then apply similar concept of one hot encoding. For the future work, this method could be useful.

### 2) Another model

- We can try “Alternating Least Squares Method for Recommendation” with stochastic gradient descent (SGD) optimizer. This model is known as pretty effective algorithm for recommender systems. It just needs rating matrix from ‘m’ users and ‘n’ items. Then, this model will try to learn a matrix of factors (features) which represent items.

- In our problem, we can regard each track as ‘item’ and each playlist as ‘user’. If a track is included in a playlist, we can put high ratings, and if not, we can put low ratings. Furthermore, by using ‘related artist’ and ‘top tracks’ of each artist information, we can elaborate this rating matrix more sophisticatedly. For the future work, we can try this model and compare the performance with auto encoder model.

### 3) Hyper parameter tuning (with cross validation)

- We chose our final model as an auto encoder. However, there are a lot of hyperparameters that we can tune (architecture of the neural network, activation function, loss function, optimizer, # of epoch, etc.). We ended up using our default choice for the final model, but if we can do hyper parameter tuning with cross validation, the performance of our model would be improved a lot.

### 4) Different K values - model

- In this project, we used K = 25, N = 55 only. However, we could train our autoencoder model on different K values. Maybe, we can see that just simple average model would outperform than the auto encoder model for the different K values. Comparing the performance between autoencoder model and simple average model on different K values could be an interesting thing for the future work.

- Furthermore, for the future work, we can try K = 0 case (Cold Start Problem) as well. In this case, we need to train our model only based on the playlist's metadata. Thus, it will be pretty difficult problem.