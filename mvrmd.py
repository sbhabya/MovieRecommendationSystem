from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
from time import time
import math
import pandas as pd
from pyspark.sql import SparkSession

conf = SparkConf().setAppName('Movie Recommender').set(
    'spark.driver.memory', '6G').set(
        'spark.executor.memory', '4G').set(
            'spark.python.worker.memory', '4G')
spark = SparkSession.builder.appName("DataProcExample").getOrCreate()
sc = spark.sparkContext

#creating RDD for movies and rating from files stored in hdfs
movies_df = spark.read.text('hdfs:///user/bhabyasinha_97/movies.csv')
ratings_df = spark.read.text('hdfs:///user/bhabyasinha_97/ratings.csv')

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

def run_als_iter(reg_param, rank, seed, als_num_iters):
    model = ALS.train(training_RDD, rank, seed=seed, iterations=als_num_iters,lambda_=reg_param)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error_validation = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    # print("For validation data the RMSE is ", error_validation)
    return error_validation

def gen_hyperparams():
    rank_arr = [4,6,8,10,12,14]
    reg_param_arr = [0.00555556, 0.01111111, 0.01666667, 0.02222222, 0.02777778, 0.03333333, 0.03888889, 0.04444444, 0.05]
    return rank_arr, reg_param_arr

def run_hyperparam_tuning(seed, als_num_iters):
    rank_arr, reg_param_arr = gen_hyperparams()
    error_df = pd.DataFrame(index=rank_arr, columns=reg_param_arr)
    for rank_ in rank_arr:
        for reg_param_ in reg_param_arr:
            err_ = run_als_iter(reg_param_,rank_,seed,als_num_iters)
            error_df.loc[rank_, reg_param_] = err_
            print(f"rank: {rank_} | reg_param: {reg_param_} | error: {err_}")
    error_df.to_csv("/home/bhabyasinha_97/error_vs_hyperparams.csv", index = False)

    return error_df

def optimize_num_iters():
    num_iters_arr = [10,12,14,16,18,20,22,24]
    seed = 5
    error_df = pd.DataFrame(index=num_iters_arr, columns=["error"])
    for num_iters_ in num_iters_arr:
        err_ = run_als_iter(0.1,12,seed,num_iters_)
        print(f"{num_iters_} iters: {err_} error")
        error_df.loc[num_iters_, "error"] = err_
    error_df.to_csv("/home/bhabyasinha_97/error_vs_num_iters.csv", index = False)
    return error_df

if __name__ == "__main__":
    # optimize_num_iters()
    run_hyperparam_tuning(5,10)   
