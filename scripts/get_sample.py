from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("Get sample")  # type: ignore
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)

weights = {
    "digital_ebook_purchase": 0.009130269933600202,
    "grocery": 0.009130269933600204,
    "health_personal_care": 0.009130269933600204,
    "music": 0.009130269933600204,
    "office_products": 0.009130269933600202,
    "pc": 0.009130269933600204,
    "pet_products": 0.009130269933600204,
    "sports": 0.009130269933600204,
    "tools": 0.009130269933600202,
    "toys": 0.009130269933600204,
    "video_dvd": 0.009130269933600202,
    "video": 0.009130269933600204,
    "wireless": 0.009130269933600204,
    "multilingual": 0.009130269933600204,
    "shoes": 0.009130269933600204,
    "digital_software": 0.009130269933600204,
    "books": 0.009130269933600202,
    "apparel": 0.009130269933600204,
    "automotive": 0.009130269933600202,
    "baby": 0.009130269933600204,
    "beauty": 0.009130269933600202,
    "camera": 0.009130269933600204,
    "digital_music_purchase": 0.009130269933600204,
    "digital_video_download": 0.009130269933600202,
    "electronics": 0.009130269933600202,
    "furniture": 0.009130269933600204,
    "outdoors": 0.009130269933600204,
    "video_games": 0.009130269933600202,
    "mobile_apps": 0.009130269933600204,
    "mobile_electronics": 0.009130269933600202,
    "musical_instruments": 0.009130269933600204,
    "digital_video_games": 0.009130269933600204,
    "watches": 0.009130269933600202,
    "software": 0.009130269933600204,
    "major_appliances": 0.009130269933600204,
    "gift_card": 0.009130269933600204,
    "personal_care_appliances": 0.009130269933600202,
}

if __name__ == "__main__":
    data = spark.read.csv(
        "s3a://amazon-reviews-eafit/data/*.tsv", sep=r"\t", header=True
    )

    selected_data = data.select(
        "customer_id",
        "product_id",
        "star_rating",
        "category",
        "review_date",
        "verified_purchase",
    )

    filtered_data = (
        selected_data.groupBy("customer_id").count().filter(F.col("count") >= 3)
    )

    filtered_data = selected_data.join(filtered_data, "customer_id", "inner").drop(
        "count"
    )

    sampled_data = filtered_data.dropna()

    sampled_data = filtered_data.filter(data.verified_purchase == True)

    sampled_data = sampled_data.withColumn(
        "star_rating", sampled_data["star_rating"].cast("float")
    )

    sampled_data = sampled_data.withColumn(
        "customer_id", sampled_data["customer_id"].cast("int")
    )

    sampled_data = data.sampleBy("category", fractions=weights)

    sampled_data.write.parquet("s3a://amazon-reviews-eafit/sample", mode="overwrite")

    # Stop SparkSession
    spark.stop()
