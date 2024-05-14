import os

import streamlit as st
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("Streamlit App")  # type: ignore
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
    .config("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
    .config(
        "fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    )
    .getOrCreate()
)
model = ALSModel.load(
    f"s3a://{os.getenv('BUCKET_NAME')}/model-random-stratified-split-sample"
)


def main():
    st.title("Sistema de recomendación para Amazon")

    # Get the input ID
    id_input = st.text_input("Introduce el ID del usuario:")

    if id_input:
        with st.spinner("Por favor espere..."):
            customer_id = int(id_input)
            df = spark.createDataFrame([(customer_id,)], ["customer_id"])
            recommendations = model.recommendForUserSubset(df, numItems=10)

            if recommendations.count() == 0:
                st.warning("No se encontraron recomendaciones para este usuario.")
            else:
                recommendations_for_user = recommendations.select(
                    "recommendations"
                ).collect()
                st.subheader(f"Recomendaciones para el usuario {customer_id}:")
                recommendation_items = recommendations_for_user[0].recommendations
                predictions_df = spark.createDataFrame(recommendation_items).toPandas()
                st.dataframe(predictions_df)

    else:
        st.warning("Por favor introduce un ID.")


if __name__ == "__main__":
    main()
