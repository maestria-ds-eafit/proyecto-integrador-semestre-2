from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("EDA")  # type: ignore
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)

if __name__ == "__main__":
    verified_data = spark.read.parquet("s3a://amazon-reviews-eafit/refined/")

    verified_records = verified_data.count()

    # 12. Get the average rating per category and save this to a variable
    avg_rating_per_category = verified_data.groupBy("category").agg(
        F.avg("star_rating").alias("avg_rating")
    )

    # 13. Count the distinct products and save it to a variable
    distinct_products_count = verified_data.select("product_id").distinct().count()

    # 14. Count the distinct customers and save it to a variable
    distinct_customers_count = verified_data.select("customer_id").distinct().count()

    # 15. Get the median of the star_rating column and save it to variable
    avg_rating = verified_data.agg(F.avg("star_rating")).collect()[0][0]

    # 16. Get the average of reviews per user and save this value to a variable
    avg_reviews_per_user = verified_records / distinct_customers_count

    # 17. Get the average of reviews per product and save this value to a variable
    avg_reviews_per_product = verified_records / distinct_products_count

    # 18. Get number of records per category
    records_per_category = verified_data.groupBy("category").count()

    summary_statistics = spark.createDataFrame(
        [
            Row(metric="Distinct Products Count", value=float(distinct_products_count)),
            Row(
                metric="Distinct Customers Count", value=float(distinct_customers_count)
            ),
            Row(metric="Average Rating", value=float(avg_rating)),
            Row(metric="Average Reviews per User", value=float(avg_reviews_per_user)),
            Row(
                metric="Average Reviews per Product",
                value=float(avg_reviews_per_product),
            ),
        ]
    )

    (
        summary_statistics.coalesce(1)  # Save as a single CSV file
        .write.mode("overwrite")
        .option("header", "true")
        .csv("s3a://amazon-reviews-eafit/eda/summary_statistics")
    )

    (
        avg_rating_per_category.coalesce(1)  # Save as a single CSV file
        .write.mode("overwrite")
        .option("header", "true")
        .csv("s3a://amazon-reviews-eafit/eda/avg_rating_per_category")
    )

    (
        records_per_category.coalesce(1)  # Save as a single CSV file
        .write.mode("overwrite")
        .option("header", "true")
        .csv("s3a://amazon-reviews-eafit/eda/records_per_category")
    )

    # Stop SparkSession
    spark.stop()
