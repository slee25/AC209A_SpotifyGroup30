---
title: Results and Conclusion
notebook: 
nav_include: 6
---


## Contents
{:.no_toc}
*  
{: toc}


## Conclusion

here goes Results

## Discussion and Future Work

### 1) Feature engineering - ‘related artist’

- In this project, we couldn't find a good way to incorporate 'related artists' (of the artist) information in a DataFrame structure. This information is definitely useful to find a connection between the tracks, by checking artists of them. The problem is that this is categorical features with too many categories (=artists). Thus, if we just incorporate this using similar concept of 'one-hot encoding', then it will adds a ton of features. 

- Possibly, “connecting graph” approach could be a solution for this problem. The nodes of connecting graph are artists, and two nodes are connected only if one artist is listed as 'related artist' of the other artist. Then, dense connections mean that all these authors in these dense clusters are closely related, so we can assign a single category for all these authors. By splitting the graph at the sparse connection, we could reduce the number of categories as a reasonable manner. Instead of directly using one hot encoding, we can reduce the number of categories first and then apply similar concept of one hot encoding. For the future work, this method could be useful.

### 2) Another model

- We can try “Alternating Least Squares Method for Recommendation” with stochastic gradient descent (SGD) optimizer. This model is known as pretty effective algorithm for recommender systems. It just needs rating matrix from ‘m’ users and ‘n’ items. Then, this model will try to learn a matrix of factors (features) which represent items.

- In our problem, we can regard each track as ‘item’ and each playlist as ‘user’. If a track is included in a playlist, we can put high ratings, and if not, we can put low ratings. Furthermore, by using ‘related artist’ and ‘top tracks’ of each artist information, we can elaborate this rating matrix more sophisticatedly. For the future work, we can try this model and compare the performance with auto encoder model.

### 3) Hyper parameter tuning (with cross validation)

- We chose our final model as an auto encoder. However, there are a lot of hyperparameters that we can tune (architecture of the neural network, activation function, loss function, optimizer, # of epoch, etc.). We ended up using our default choice for the final model, but if we can do hyper parameter tuning with cross validation, the performance of our model would be improved a lot.