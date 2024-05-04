from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession

spark = (
    SparkSession.builder.appName("Collaborative Filtering")  # type: ignore
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)

if __name__ == "__main__":
    data = spark.read.csv(
        "s3a://amazon-reviews-eafit/refined/*.csv", sep=r"\t", header=True
    )

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
    )
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="star_rating", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error = {rmse}")

    # Generate top 10 product recommendations for each user
    userRecs = model.recommendForAllUsers(10)

    # Show top 10 recommendations for each user
    print(userRecs.show())

    summary = spark.createDataFrame(
        [
            Row(metric="RMSE", value=rmse),
        ]
    )

    (
        summary.coalesce(1)  # Save as a single CSV file
        .write.mode("overwrite")
        .option("header", "true")
        .csv("s3a://amazon-reviews-eafit/rmse")
    )

    # Stop SparkSession
    spark.stop()
