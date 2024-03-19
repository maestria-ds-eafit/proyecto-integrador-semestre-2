from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Collaborative Filtering").getOrCreate()  # type: ignore
data = spark.read.option("header", "true").csv("s3://amazon-reviews-eafit/*.csv")

# Convert relevant columns to integer type
data = (
    data.withColumn("customer_id", data["customer_id"].cast("int"))
    .withColumn("product_id", data["product_id"].cast("int"))
    .withColumn("star_rating", data["star_rating"].cast("float"))
)

# Split data into training and test sets
(training, test) = data.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(
    maxIter=5,
    regParam=0.01,
    userCol="customer_id",
    itemCol="product_id",
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

# Stop SparkSession
spark.stop()
