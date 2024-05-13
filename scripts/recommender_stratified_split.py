import argparse

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

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
    SparkSession.builder.appName("Collaborative Filtering with stratified split")  # type: ignore
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)


def get_metrics(model, dataset):
    predictions = model.transform(dataset)
    evaluator_rmse = RegressionEvaluator(
        metricName="rmse", labelCol="star_rating", predictionCol="prediction"
    )
    evaluator_mae = RegressionEvaluator(
        metricName="mae", labelCol="star_rating", predictionCol="prediction"
    )
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    predictions_count = predictions.count()
    return rmse, mae, predictions_count


def split_data(data):
    windowSpec = Window.partitionBy("customer_id").orderBy(col("review_date").desc())
    data = data.withColumn("index", row_number().over(windowSpec))

    training = data.where(col("index") >= 3)
    validation = data.where(col("index") == 2)
    test = data.where(col("index") <= 1)

    return training, validation, test


if __name__ == "__main__":
    data_path = f"s3a://amazon-reviews-eafit/{'sample-for-model' if use_sampling else 'refined'}/"
    data = spark.read.parquet(data_path)

    indexer = StringIndexer(inputCol="product_id", outputCol="item_id")

    data = indexer.fit(data).transform(data)

    training, validation, test = split_data(data)

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
    rmse_test, mae_test, predictions_test_count = get_metrics(model, test)

    # Descomentar cuando sepamos cómo hacer la validación en toda la data
    # rmse_validation, mae_validation, predictions_validation_count = get_metrics(model, validation)

    # print(f"Predictions count (validation): {predictions_validation_count}")
    # print(f"RMSE (validation) = {rmse_validation}")
    # print(f"MAE (validation) = {mae_validation}")

    print(f"Predictions count (test): {predictions_test_count}")
    print(f"RMSE (test) = {rmse_test}")
    print(f"MAE (test) = {mae_test}")

    summary = spark.createDataFrame(
        [
            # Descomentar cuando sepamos cómo hacer la validación en toda la data
            # Row(
            #     metric="Predictions count (validation)",
            #     value=float(predictions_validation_count),
            # ),
            # Row(metric="RMSE (validation)", value=float(rmse_validation)),
            # Row(metric="MAE (validation)", value=float(mae_validation)),
            Row(
                metric="Predictions count (test)",
                value=float(predictions_test_count),
            ),
            Row(metric="RMSE (test)", value=float(rmse_test)),
            Row(metric="MAE (test)", value=float(mae_test)),
        ]
    )

    summary_path = f"s3a://amazon-reviews-eafit/{'rmse-stratified-split-sample' if use_sampling else 'rmse-stratified-split'}"

    (
        summary.coalesce(1)  # Save as a single CSV file
        .write.mode("overwrite")
        .option("header", "true")
        .csv(summary_path)
    )

    # Stop SparkSession
    spark.stop()
