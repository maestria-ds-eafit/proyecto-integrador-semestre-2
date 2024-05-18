from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("Get Reviews Count Distribution")  # type: ignore
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)


def get_count_of_customers_by_reviews(data):
    reviews_per_user = data.groupBy("customer_id").agg(
        F.count("star_rating").alias("review_count")
    )

    result = reviews_per_user.groupBy("review_count").agg(
        F.count("customer_id").alias("user_count")
    )

    return result


if __name__ == "__main__":
    data = spark.read.csv(
        "s3://amazon-reviews-eafit/data/*.tsv", sep=r"\t", header=True
    )

    data = data.select(
        "customer_id",
        "product_id",
        "star_rating",
        "category",
        "review_date",
        "verified_purchase",
        "review_id",
        "product_title",
        "product_category",
    )

    data = data.dropna()

    data = data.filter(data.verified_purchase == True)

    count_of_customers_by_reviews = get_count_of_customers_by_reviews(data)

    count_of_customers_by_reviews.write.mode("overwrite").parquet(
        "s3://amazon-reviews-eafit/eda/count_of_customers_by_reviews"
    )

    # Stop SparkSession
    spark.stop()
