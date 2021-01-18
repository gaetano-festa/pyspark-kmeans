#!/usr/bin/env python3

from pyspark.sql import SparkSession

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DateType, TimestampType, IntegerType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

import os
import shutil
import csv

spark = SparkSession.builder.appName('Customer Segmentation').getOrCreate()

sc = spark.sparkContext

# Read data from csv files

# To properly 
users_schema = StructType([
    StructField('timestamp', TimestampType(), nullable=False),
    StructField('userId', IntegerType(), False),
    StructField('nick', StringType(), False),
    StructField('twitter', StringType(), False),
    StructField('dob', DateType(), False),
    StructField('country', StringType(), False)
])

users = spark.read.load('data/users.csv', format='csv',schema=users_schema, header=True)

#users.printSchema()

#print('Latest date:')
users.select(F.to_date(F.max(F.date_trunc('day', 'timestamp')))).show()


users = users.withColumn('age', F.datediff(F.to_date(F.lit('2016-06-16'), 'yyyy-mm-dd'), users.dob)/365)

ages = users.select(['userId', 'age'])
#ages.show()


buy_click = spark.read.load('data/buy-clicks.csv', format='csv',inferSchema=True, header=True, timestampFormat="yyyy-MM-dd HH:mm:ss")

#buy_click.printSchema()


revenue_by_session_by_user = buy_click.groupBy(['userId', 'userSessionId']).agg(F.sum('price').alias('revenue')).select(['userId', 'revenue'])
revenues=revenue_by_session_by_user.groupBy('userId').agg(F.mean('revenue').alias('avg_buy'), F.min('revenue').alias('min_buy'), F.max('revenue').alias('max_buy'))
#revenues.show()


game_click = spark.read.load('data/game-clicks.csv', format='csv',inferSchema=True, header=True, timestampFormat="yyyy-MM-dd HH:mm:ss")

#game_click.printSchema()


avg_ishit = game_click.groupBy(['userId']).agg(F.mean('isHit').alias('avg_isHit'))

user_session = spark.read.load('data/user-session.csv', format='csv',inferSchema=True, header=True, timestampFormat="yyyy-MM-dd HH:mm:ss")

#user_session.printSchema()

team = spark.read.load('data/team.csv', format='csv',inferSchema=True, header=True, timestampFormat="yyyy-MM-dd HH:mm:ss")
#team.printSchema()


strengths = team.join(user_session, on='teamId', how='inner').select(['userId', 'strength']).dropDuplicates()

data = ages.join(revenues, on='userId', how='inner').join(avg_ishit, on='userId', how='inner').join(strengths, on='userId', how='left').na.fill(0)
#data.show()


data.describe().toPandas().transpose()




data = data.withColumn('log_age', F.log('age')).withColumn('log_avg_buy', F.log('avg_buy'))\
    .withColumn('log_min_buy', F.log('min_buy')).withColumn('log_max_buy', F.log('max_buy'))

feature = data.columns
feature.remove('userId')
features = feature[4:]

assembler = VectorAssembler(inputCols=features, outputCol='features_unscaled')
assembled = assembler.transform(data)

scaler = StandardScaler(inputCol='features_unscaled', outputCol='features', withStd=True, withMean=True)
scaler_model = scaler.fit(assembled)
scaled_data = scaler_model.transform(assembled)

scaled_data = scaled_data.select('features')
scaled_data = scaled_data.coalesce(4)

evaluator = ClusteringEvaluator()

silhuette_scores = {}
centers = {}

tmp_models_dir = 'tmp_models'
model_summaries_file = os.path.join(tmp_models_dir, 'model_summaries.txt')

# If the temporary directory already exists it will be removed to create a fresh one.
# Other managements of this case are possible but they won't be considered here.
if os.path.exists(tmp_models_dir):
    shutil.rmtree(tmp_models_dir)

os.mkdir(tmp_models_dir)

for k in range(2,4):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol('features')
    model = kmeans.fit(scaled_data)
    transformed = model.transform(scaled_data)
    silhuette_scores[k] = evaluator.evaluate(transformed)
    centers[k] = model.clusterCenters()
    model.save(os.path.join(tmp_models_dir, "model_w_k_{}".format(k)))

# Save the relevant results in a file to choose the optimal number of clusters
output_file = 'clustering_results.csv'

# If the file already exists will be overwritten
with open(output_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['k'] + ['score'] + list(features))

    for k, v in centers.items():
        centers_list = [c.tolist() for c in v]
        for center in centers_list:
            writer.writerow([k] + [silhuette_scores[k]] + center)
