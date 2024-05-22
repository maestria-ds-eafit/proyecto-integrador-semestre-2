import argparse

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col, expr, rank
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
    SparkSession.builder.appName("Collaborative Filtering with random stratified split")  # type: ignore
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)


def split_data(data, percent_items_to_mask=0.2):
    user_window = Window.partitionBy("customer_id").orderBy(col("product_id").desc())
    data_processed = data.withColumn(
        "number_of_products", expr("count(*) over (partition by customer_id)")
    )
    data_processed = data_processed.withColumn(
        "number_of_products_to_mask",
        (col("number_of_products") * percent_items_to_mask).cast("int"),
    )
    data_processed = data_processed.withColumn("product_rank", rank().over(user_window))

    training = data_processed.filter(
        col("product_rank") > col("number_of_products_to_mask")
    )
    test = data_processed.filter(
        col("product_rank") <= col("number_of_products_to_mask")
    )

    return training, test


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


def get_string_indexer(data):
    string_indexer = StringIndexer(inputCol="product_id", outputCol="item_id")
    string_indexer_model = string_indexer.fit(data)

    return string_indexer_model


def save_string_indexer_inverter(string_indexer_model):
    if not use_sampling:
        return

    inverter = IndexToString(
        inputCol="item_id",
        outputCol="original_item_id",
        labels=string_indexer_model.labels,
    )

    inverter.write().overwrite().save(
        f"s3://amazon-reviews-eafit/{'inverter-random-stratified-split-sample' if use_sampling else 'inverter-random-stratified-split'}"
    )


if __name__ == "__main__":
    data_path = (
        f"s3://amazon-reviews-eafit/{'sample-for-demo' if use_sampling else 'refined'}/"
    )
    data = spark.read.parquet(data_path)

    string_indexer_model = get_string_indexer(data)
    save_string_indexer_inverter(string_indexer_model)

    data = string_indexer_model.transform(data)

    training, test = split_data(data, percent_items_to_mask=0.3)

    if use_sampling:
        training.write.mode("overwrite").parquet(
            f"s3://amazon-reviews-eafit/sample-training"
        )
        test.write.mode("overwrite").parquet(f"s3://amazon-reviews-eafit/sample-test")

    # Build the recommendation model using ALS on the training data
    als = ALS(
        maxIter=15,
        regParam=0.1,
        rank=15,
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

    print(f"Predictions count (test): {predictions_test_count}")
    print(f"RMSE (test) = {rmse_test}")
    print(f"MAE (test) = {mae_test}")

    summary = spark.createDataFrame(
        [
            Row(
                metric="Predictions count (test)",
                value=float(predictions_test_count),
            ),
            Row(metric="RMSE (test)", value=float(rmse_test)),
            Row(metric="MAE (test)", value=float(mae_test)),
        ]
    )

    summary_path = f"s3://amazon-reviews-eafit/{'rmse-random-stratified-split-sample' if use_sampling else 'rmse-random-stratified-split'}"

    (
        summary.coalesce(1)  # Save as a single CSV file
        .write.mode("overwrite")
        .option("header", "true")
        .csv(summary_path)
    )

    # Save the model to S3
    if use_sampling:
        model_path = f"s3://amazon-reviews-eafit/model-random-stratified-split-sample"
        model.write().overwrite().save(model_path)

    # Stop SparkSession
    spark.stop()
