from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import os
import shutil
import csv

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructField, StructType, StringType, DoubleType
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import *


def csv_line(data):
    r = ','.join(str(d) for d in data[1])
    return str(data[0]) + "," + r


def main():
    spark = SparkSession.builder.appName("PySparkAbalone").getOrCreate()
    
    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    spark.sparkContext._jsc.hadoopConfiguration().set("mapred.output.committer.class",
                                                      "org.apache.hadoop.mapred.FileOutputCommitter")
    
    # Defining the schema corresponding to the input data. The input data does not contain the headers
    schema = StructType([StructField("sex", StringType(), True), 
                         StructField("length", DoubleType(), True),
                         StructField("diameter", DoubleType(), True),
                         StructField("height", DoubleType(), True),
                         StructField("whole_weight", DoubleType(), True),
                         StructField("shucked_weight", DoubleType(), True),
                         StructField("viscera_weight", DoubleType(), True), 
                         StructField("shell_weight", DoubleType(), True), 
                         StructField("rings", DoubleType(), True)])

    # Downloading the data from S3 into a Dataframe
    total_df = spark.read.csv(('s3a://' + os.path.join(args['s3_input_bucket'], args['s3_input_key_prefix'],
                                                   'abalone.csv')), header=False, schema=schema)

    #StringIndexer on the sex column which has categorical value
    sex_indexer = StringIndexer(inputCol="sex", outputCol="indexed_sex")
    
    #one-hot-encoding is being performed on the string-indexed sex column (indexed_sex)
    sex_encoder = OneHotEncoder(inputCol="indexed_sex", outputCol="sex_vec")

    #vector-assembler will bring all the features to a 1D vector for us to save easily into CSV format
    assembler = VectorAssembler(inputCols=["sex_vec", 
                                           "length", 
                                           "diameter", 
                                           "height", 
                                           "whole_weight", 
                                           "shucked_weight", 
                                           "viscera_weight", 
                                           "shell_weight"], 
                                outputCol="features")
    
    # The pipeline comprises of the steps added above
    pipeline = Pipeline(stages=[sex_indexer, sex_encoder, assembler])
    
    # This step trains the feature transformers
    model = pipeline.fit(total_df)
    
    # This step transforms the dataset with information obtained from the previous fit
    transformed_total_df = model.transform(total_df)
    
    # Split the overall dataset into 80-20 training and validation
    (train_df, validation_df) = transformed_total_df.randomSplit([0.8, 0.2])
    
    # Convert the train dataframe to RDD to save in CSV format and upload to S3
    train_rdd = train_df.rdd.map(lambda x: (x.rings, x.features))
    train_lines = train_rdd.map(csv_line)
    train_lines.saveAsTextFile('s3a://' + os.path.join(args['s3_output_bucket'], args['s3_output_key_prefix'], 'train'))
    
    # Convert the validation dataframe to RDD to save in CSV format and upload to S3
    validation_rdd = validation_df.rdd.map(lambda x: (x.rings, x.features))
    validation_lines = validation_rdd.map(csv_line)
    validation_lines.saveAsTextFile('s3a://' + os.path.join(args['s3_output_bucket'], args['s3_output_key_prefix'], 'validation'))


if __name__ == "__main__":
    main()