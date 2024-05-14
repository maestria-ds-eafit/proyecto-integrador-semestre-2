from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = (
    SparkSession.builder.appName("Get sample for model")  # type: ignore
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)

if __name__ == "__main__":
    data = spark.read.parquet("s3://amazon-reviews-eafit/refined/")

    distinct_customer_ids = data.select("customer_id").distinct()

    sampled_customer_ids = distinct_customer_ids.sample(False, 0.1)
    sampled_customer_ids = sampled_customer_ids.limit(500)

    list_of_sampled_ids = [row["customer_id"] for row in sampled_customer_ids.collect()]

    data = data.filter(col("customer_id").isin(list_of_sampled_ids))

    data.write.parquet("s3://amazon-reviews-eafit/sample-for-model", mode="overwrite")

    spark.stop()
