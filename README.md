# MovieRecommendationSystem

Movie Recommendation System uses Alternating Least Squares (ALS) Machine Learning classifier to provide movie recommendations to a user depending on the user's ratings of other movies, and also depending on all the ratings of other users. 

* Model is trained on a large 25 million rating dataset, (MovieLens)[https://grouplens.org/datasets/movielens/25m/, "MovieLens"].
* Apache Spark and HDFS are used to handle the large size of data
* Google Dataproc is used to run Apache Spark and Hadoop on the cloud

## Hyperparameter tuning and iteration convergence

* ALS objective-function minimization is truncated at a finite number of iterations. We sweep this iteration number to test for convergence.

![Convergence with respect to number of iterations in ALS optimization](./Images/iters_vs_rmse.png)

  
