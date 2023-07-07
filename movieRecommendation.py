from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
from time import time
import math
from pyspark.sql import SparkSession

conf = SparkConf().setAppName('Movie Recommender').set(
    'spark.driver.memory', '6G').set(
        'spark.executor.memory', '4G').set(
            'spark.python.worker.memory', '4G')
spark = SparkSession.builder.appName("DataProcExample").getOrCreate()
sc = spark.sparkContext

#creating RDD for movies and rating from files stored in hdfs
movies_df = spark.read.text('hdfs:///user/bhabyasinha_97/data/movies.csv')
ratings_df = spark.read.text('hdfs:///user/bhabyasinha_97/data/ratings.csv')

#extracting the header 
movies_header = movies_df.first()
ratings_header = ratings_df.first()

#removing the header from the rdds
movies_rdd = movies_df.filter(movies_df.value!= movies_header.value).rdd.map(
    lambda line: line[0].split(",")).map(
        lambda tokens: (tokens[0],tokens[1])).cache()
ratings_rdd = ratings_df.filter(ratings_df.value != ratings_header.value).rdd.map(
    lambda line: line[0].split(",")).map(
        lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()    


training_RDD, validation_RDD, test_RDD = ratings_rdd.randomSplit([6, 2, 2], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02
min_error = float('inf')
best_rank = -1
best_iteration = -1
min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print("For rank " + str(rank) + "the RMSE is " + str(error))
    if error < min_error:
        min_error = error
        best_rank = rank

print("The best model was trained with rank ",best_rank)


training_Ratings_RDD, test_Ratings_RDD = ratings_rdd.randomSplit([7, 3], seed=0)
complete_model = ALS.train(training_Ratings_RDD, best_rank, seed=seed, 
                           iterations=iterations, lambda_=regularization_parameter)
test_for_predict_RDD = test_Ratings_RDD.map(lambda x: (x[0], x[1]))

predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print("For testing data the RMSE is ",error)

complete_movies_titles = movies_rdd.map(lambda x: (int(x[0]),x[1]))
print("Total number of movies in the complete dataset" , complete_movies_titles.count())

def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    ratings = [float(x) for x in ID_and_ratings_tuple[1]]  # Convert ratings to floats
    return ID_and_ratings_tuple[0], (nratings, sum(ratings) / nratings)


movie_ID_with_ratings_RDD = (ratings_rdd.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,9), # Star Wars (1977)
     (0,1,8), # Toy Story (1995)
     (0,16,7), # Casino (1995)
     (0,25,8), # Leaving Las Vegas (1995)
     (0,32,9), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,4), # Flintstones, The (1994)
     (0,379,3), # Timecop (1994)
     (0,296,7), # Pulp Fiction (1994)
     (0,858,10) , # Godfather, The (1972)
     (0,50,8) # Usual Suspects, The (1995)
    ]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
complete_data_with_new_ratings_RDD = ratings_rdd.union(new_user_ratings_RDD)

t0 = time()
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed, 
                              iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0
print ("New model trained in ", round(tt,3))
new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) # get just movie IDs

# keep just those not on the ID list 
new_user_unrated_movies_RDD = (movies_rdd.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)
new_user_unrated_movies_RDD = (movies_rdd.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)

new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
new_user_recommendations_rating_title_and_count_RDD.take(3)
new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])
print ('TOP recommended movies (with more than 25 reviews):\n','\n'.join(map(str, top_movies)))
