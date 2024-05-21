import itertools

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col, expr, rank
from pyspark.sql.window import Window

spark = (
    SparkSession.builder.appName("Hyperparameter tuning")  # type: ignore
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
    print("Predictions count: ", predictions_count)
    print("RMSE: ", rmse)
    print("MAE: ", mae)
    return rmse, mae, predictions_count


def get_string_indexer(data):
    string_indexer = StringIndexer(inputCol="product_id", outputCol="item_id")
    string_indexer_model = string_indexer.fit(data)

    return string_indexer_model


def train_model(maxIter=5, regParam=0.1, rank=10):
    als = ALS(
        maxIter=maxIter,
        regParam=regParam,
        userCol="customer_id",
        itemCol="item_id",
        ratingCol="star_rating",
        seed=42,
        nonnegative=True,
        rank=rank,
        coldStartStrategy="drop",
    )
    model = als.fit(training)
    return model


if __name__ == "__main__":
    data_path = f"s3://amazon-reviews-eafit/sample-for-model"
    data = spark.read.parquet(data_path)

    string_indexer_model = get_string_indexer(data)

    data = string_indexer_model.transform(data)

    training, test = split_data(data, percent_items_to_mask=0.3)
    training, validation = split_data(training, percent_items_to_mask=0.3)

    # parameters = {
    #     "maxIter": [5, 10, 15],
    #     "regParam": [0.001, 0.01, 0.1],
    #     "rank": [1, 5, 10, 15, 20],
    # }
    parameters = {
        "maxIter": [5, 10, 15],
        "regParam": [0.1, 0.01, 0.001],
        "rank": [5, 10, 15],
    }

    param_combinations = list(itertools.product(*parameters.values()))
    tuning_parameters = [
        {"maxIter": maxIter, "regParam": regParam, "rank": rank}
        for maxIter, regParam, rank in param_combinations
    ]

    corresponding_rmse, best_mae, best_predictions_count, best_parameters = (
        float("inf"),
        float("inf"),
        0,
        None,
    )

    for parameters_combination in tuning_parameters:
        print(f"Parameters: {parameters_combination}")
        model = train_model(**parameters_combination)
        rmse, mae, predictions_count = get_metrics(model, validation)
        print("-----------------------------------------")
        if mae < best_mae:
            best_mae = mae
            corresponding_rmse = rmse
            best_parameters = parameters_combination
            best_predictions_count = predictions_count

    print(f"Best parameters: {best_parameters}")
    print(f"Best MAE: {mae}")
    print(f"RMSE corresponding to the best MAE: {corresponding_rmse}")

    summary = spark.createDataFrame(
        [
            Row(
                metric="Predictions count (validation)",
                value=float(best_predictions_count),
            ),
            Row(metric="RMSE (validation)", value=float(corresponding_rmse)),
            Row(metric="MAE (validation)", value=float(best_mae)),
            Row(metric="Best parameters", value=str(best_parameters)),
        ]
    )

    summary_path = f"s3://amazon-reviews-eafit/hyperparameter-tuning"

    (
        summary.coalesce(1)  # Save as a single CSV file
        .write.mode("overwrite")
        .option("header", "true")
        .csv(summary_path)
    )

    # Stop SparkSession
    spark.stop()
