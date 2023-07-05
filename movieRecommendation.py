from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
import math

conf = SparkConf().setAppName('Movie Recommender').set(
    'spark.driver.memory', '6G').set(
        'spark.executor.memory', '4G').set(
            'spark.python.worker.memory', '4G')
sc = SparkContext.getOrCreate(conf=conf)
movies_df = spark.read.text('hdfs:///user/bhabyasinha_97/movies.csv')
ratings_df = spark.read.text('hdfs:///user/bhabyasinha_97/ratings.csv')
movies_header = movies_df.first()
ratings_header = ratings_df.first()
movies_rdd = movies_df.filter(movies_df.value != movies_header.value).rdd.map(
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
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print("For rank %s the RMSE is %s",rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank
print("The best model was trained with rank %s",best_rank)
model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print("For testing data the RMSE is %s", error)
training_Ratings_RDD, test_Ratings_RDD = ratings_rdd.randomSplit([7, 3], seed=0)

complete_model = ALS.train(training_Ratings_RDD, best_rank, seed=seed, 
                           iterations=iterations, lambda_=regularization_parameter)

test_for_predict_RDD = test_Ratings_RDD.map(lambda x: (x[0], x[1]))

predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    
print("For testing data the RMSE is %s",error)
