from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("Calculate rows per category")  # type: ignore
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
        "s3a://amazon-reviews-eafit/data/*.tsv", sep=r"\t", header=True
    )

    # Perform groupBy operation on the "category" column
    grouped_data = data.groupBy("category").count()

    # Save the result to a CSV file in S3
    grouped_data.write.csv(
        "s3a://amazon-reviews-eafit/output/counts_per_category",
        mode="overwrite",
        header=True,
    )

    # Stop SparkSession
    spark.stop()
