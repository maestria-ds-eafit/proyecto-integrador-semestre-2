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
    data = spark.read.csv(
        "s3a://amazon-reviews-eafit/data/*.tsv", sep=r"\t", header=True
    )

    # Code goes here
    # 1. Count how many records we have and save it to a variable
    total_records = data.count()

    # 2. Select the columns
    selected_data = data.select(
        "customer_id",
        "product_id",
        "star_rating",
        "category",
        "review_date",
        "verified_purchase",
    )

    # 3. Keep only those customers that appear at least 3 times
    filtered_data = (
        selected_data.groupBy("customer_id").count().filter(F.col("count") >= 3)
    )
    filtered_data = selected_data.join(filtered_data, "customer_id", "inner").drop(
        "count"
    )

    # 4. Count how many records we have after doing this filter and save it to a variable
    filtered_records = filtered_data.count()

    # 5. Delete all those records that have nulls on any of the columns
    filtered_data = filtered_data.dropna()

    # 6. Count how many records we have after doing this filter and save it to a variable
    filtered_records_no_nulls = filtered_data.count()

    # 7. Filter customers where "verified_purchase" is True
    verified_data = filtered_data.filter(filtered_data.verified_purchase == True)

    # 8. Count how many records we have after doing this filter and save it to a variable
    verified_records = verified_data.count()

    # 9. Cast "star_rating" to float
    verified_data = verified_data.withColumn(
        "star_rating", verified_data["star_rating"].cast("float")
    )

    counts_df = spark.createDataFrame(
        [
            Row(metric="Total Records", value=total_records),
            Row(metric="Filtered Records", value=filtered_records),
            Row(metric="Filtered Records (No Nulls)", value=filtered_records_no_nulls),
            Row(metric="Verified Records", value=verified_records),
        ]
    )

    (
        counts_df.coalesce(1).write.csv(  # Save as a single CSV file
            "s3a://amazon-reviews-eafit/eda/counts", mode="overwrite", header=True
        )
    )

    verified_data.write.csv(
        "s3a://amazon-reviews-eafit/refined", mode="overwrite", header=True, sep=r"\t"
    )

    # Stop SparkSession
    spark.stop()
