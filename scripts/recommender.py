import argparse

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession

parser = argparse.ArgumentParser()

parser.add_argument(
    "--use-sampling",
    type=bool,
    help="Use sampling for running the model",
    default=False,
)

# Capture args
args = parser.parse_args()
use_sampling = args.use_sampling

spark = (
    SparkSession.builder.appName("Collaborative Filtering")  # type: ignore
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)

if __name__ == "__main__":
    data_path = f"s3a://amazon-reviews-eafit/{'sample' if use_sampling else 'refined'}/"
    data = spark.read.parquet(data_path)

    indexer = StringIndexer(inputCol="product_id", outputCol="item_id")

    data = indexer.fit(data).transform(data)

    # Split data into training and test sets
    (training, test) = data.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="customer_id",
        itemCol="item_id",
        ratingCol="star_rating",
        seed=42,
        nonnegative=True,
        coldStartStrategy="drop",
    )
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator_rmse = RegressionEvaluator(
        metricName="rmse", labelCol="star_rating", predictionCol="prediction"
    )
    evaluator_mae = RegressionEvaluator(
        metricName="mae", labelCol="star_rating", predictionCol="prediction"
    )
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    predictions_count = predictions.count()

    print(f"Predictions count: {predictions_count}")
    print(f"RMSE = {rmse}")
    print(f"MAE = {mae}")

    summary = spark.createDataFrame(
        [
            Row(metric="Predictions count", value=float(predictions_count)),
            Row(metric="RMSE", value=float(rmse)),
            Row(metric="MAE", value=float(mae)),
        ]
    )

    summary_path = (
        f"s3a://amazon-reviews-eafit/{'rmse' if use_sampling else 'rmse-sample'}"
    )

    (
        summary.coalesce(1)  # Save as a single CSV file
        .write.mode("overwrite")
        .option("header", "true")
        .csv(summary_path)
    )

    # Stop SparkSession
    spark.stop()
