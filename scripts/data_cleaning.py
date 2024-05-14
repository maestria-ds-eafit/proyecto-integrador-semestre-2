from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("Data Cleaning")  # type: ignore
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

    # Count how many records we have and save it to a variable
    total_records = data.count()

    # Select the columns
    selected_data = data.select(
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

    # Delete all those records that have nulls on any of the columns
    data_without_nulls = selected_data.dropna()

    # Count how many records we have after doing this filter and save it to a variable
    data_without_nulls_count = data_without_nulls.count()

    # 7. Filter customers where "verified_purchase" is True
    verified_data = data_without_nulls.filter(
        data_without_nulls.verified_purchase == True
    )

    # 8. Count how many records we have after doing this filter and save it to a variable
    verified_data_count = verified_data.count()

    # Keep only those customers that appear at least 3 times
    customers_with_more_than_three_reviews = (
        verified_data.groupBy("customer_id").count().filter(F.col("count") >= 3)
    )

    cleaned_data = verified_data.join(
        customers_with_more_than_three_reviews, "customer_id", "inner"
    ).drop("count")

    cleaned_data_count = cleaned_data.count()

    # Cast "star_rating" to float
    cleaned_data = cleaned_data.withColumn(
        "star_rating", cleaned_data["star_rating"].cast("float")
    )

    # Cast "customer_id" to int
    cleaned_data = cleaned_data.withColumn(
        "customer_id", cleaned_data["customer_id"].cast("int")
    )

    counts_df = spark.createDataFrame(
        [
            Row(metric="Total Records", value=total_records),
            Row(metric="Filtered Records (No Nulls)", value=data_without_nulls_count),
            Row(metric="Verified Records", value=verified_data_count),
            Row(
                metric="Customers with more than three reviews count",
                value=cleaned_data_count,
            ),
        ]
    )

    counts_df.write.csv(
        "s3://amazon-reviews-eafit/eda/counts", mode="overwrite", header=True
    )

    cleaned_data.write.parquet("s3://amazon-reviews-eafit/refined", mode="overwrite")

    # Stop SparkSession
    spark.stop()
