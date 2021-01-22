from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, TimestampType, IntegerType
from pyspark.sql import functions as F
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

import os
import shutil
import csv
import sys

spark = SparkSession.builder.appName('Customer Segmentation').getOrCreate()

def load_data():
    """Load the data from the different files and join the togheter
    """

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
    #users.select(F.to_date(F.max(F.date_trunc('day', 'timestamp')))).show()
    
    
    users = users.withColumn('age', F.datediff(F.to_date(F.lit('2016-06-16'), 'yyyy-mm-dd'), users.dob)/365)
    
    ages = users.select(['userId', 'age'])
    #ages.show()

    
    buy_click = spark.read.load('data/buy-clicks.csv', format='csv',inferSchema=True, header=True, timestampFormat="yyyy-MM-dd HH:mm:ss")
    
    #buy_click.printSchema()
    
    
    revenue_by_session_by_user = buy_click.groupBy(['userId', 'userSessionId']).agg(F.sum('price').alias('revenue')).select(['userId', 'revenue'])
    revenues=revenue_by_session_by_user.groupBy('userId').agg(F.mean('revenue').alias('avg_buy'), F.min('revenue')\
            .alias('min_buy'), F.max('revenue').alias('max_buy'))
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
    
    return data


def prepare_data():
    """Commodity function to read the data from the files and prepare the features for the kmeans model fit.
    """
    # Read data from files.
    _data = load_data()

    # As the distribution of the following feature is not normal they will be log scaled to have a more
    # normally distributed distribution. This is required for kmeans algorithm to work better.
    _data = _data.withColumn('log_age', F.log('age')).withColumn('log_avg_buy', F.log('avg_buy'))\
        .withColumn('log_min_buy', F.log('min_buy')).withColumn('log_max_buy', F.log('max_buy'))

    # Select the features to use in kmeans. The features will be also standard scaled, that is mean centered
    # and scaled to have standard deviation of one.
    features = _data.columns[4:]

    assembler = VectorAssembler(inputCols=features, outputCol='features_unscaled')
    assembled = assembler.transform(_data)

    scaler = StandardScaler(inputCol='features_unscaled', outputCol='features', withStd=True, withMean=True)
    scaler_model = scaler.fit(assembled)
    scaled_data = scaler_model.transform(assembled)


    return scaled_data, features

def kmeans_scan(_data, _k_min = 2, _k_max = 6, _tmp_dir = 'tmp_models'):
    """Scan different kmeans model within the specified k range.
       The function assume that the input data are ready to be used and already contain the features column.
    """
    # Define the evaluator to find the optimal k. The evaluator compute the Siluhette score.
    evaluator = ClusteringEvaluator()

    # Dictionaries use to save the results obtained for the diferent k considered.
    silhuette_scores = {}
    centers = {}


    # If the temporary directory already exists it will be removed to create a fresh one.
    # Other managements of this case are possible but they won't be considered here, the
    # extension to these cases is straitforward.
    if os.path.exists(_tmp_dir):
        shutil.rmtree(_tmp_dir)

    os.mkdir(_tmp_dir)

    # Fit and save the model for the specifoed k
    for k in range(_k_min, _k_max + 1):
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol('features')
        model = kmeans.fit(_data)
        transformed = model.transform(_data)
        silhuette_scores[k] = evaluator.evaluate(transformed)
        centers[k] = model.clusterCenters()
        model.save(os.path.join(_tmp_dir, "model_w_k_{}".format(k)))

    return centers, silhuette_scores



def save_clustering_results(_output_file, _centers, _scores, _feature_names):
    """Save the cluster centers and Silhuette scores for the different k.
       These data will be use to choose the optimal k.
       Input:
        - output file name where the data are saved. It will be overwritten if it exists.
        - dictionary containing the cluster centers with k as key.
        - dictionary containing the Silhuette scores with l as key
        - a list with the name of the features
    """

    # If the file already exists will be overwritten
    with open(_output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['k'] + ['score'] + list(_feature_names))

        for k, v in _centers.items():
            centers_list = [c.tolist() for c in v]
            for center in centers_list:
                writer.writerow([k] + [_scores[k]] + center)



def load_kmeans_model(_model_dir):
    """Load the specified model.
    """
    if os.path.exists(_model_dir):
        print("Loading model from {} direcory...".format(_model_dir))
        model = KMeansModel.load(_model_dir)
    else:
        print('Model {} not found.'.format(_model_dir))
        sys.exit(1)

    return model

