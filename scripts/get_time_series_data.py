from pyspark.sql import SparkSession
from pyspark.sql.functions import when

spark = (
    SparkSession.builder.appName("Get time series data")  # type: ignore
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)

if __name__ == "__main__":
    data = spark.read.csv(
        "s3://amazon-reviews-eafit/data/*.tsv", sep=r"\t", header=True
    )

    data = data.dropna()

    data = data.withColumn("star_rating", data["star_rating"].cast("float"))

    data = data.withColumn(
        "sentiment",
        when(data["star_rating"] <= 2, "negative")
        .when(data["star_rating"] == 3, "neutral")
        .otherwise("positive"),
    )

    # Group by sentiment and review_date, aggregate by count
    grouped_data = data.groupBy("sentiment", "review_date").count()

    # Separate dataframes for each sentiment
    negative_df = (
        grouped_data.filter(grouped_data["sentiment"] == "negative")
        .select("review_date", "count")
        .orderBy("review_date")
    )
    neutral_df = (
        grouped_data.filter(grouped_data["sentiment"] == "neutral")
        .select("review_date", "count")
        .orderBy("review_date")
    )
    positive_df = (
        grouped_data.filter(grouped_data["sentiment"] == "positive")
        .select("review_date", "count")
        .orderBy("review_date")
    )

    negative_df.write.parquet(
        "s3://amazon-reviews-eafit/time-series-data/negative", mode="overwrite"
    )
    neutral_df.write.parquet(
        "s3://amazon-reviews-eafit/time-series-data/neutral", mode="overwrite"
    )
    positive_df.write.parquet(
        "s3://amazon-reviews-eafit/time-series-data/positive", mode="overwrite"
    )

    spark.stop()
