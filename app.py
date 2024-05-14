import os

import streamlit as st
from pyspark.ml.feature import IndexToString
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
inverter = IndexToString.load(
    f"s3a://{os.getenv('BUCKET_NAME')}/inverter-random-stratified-split-sample"
)


def main():
    st.title("Sistema de recomendación para Amazon")

    id_input = st.text_input("Introduce el ID del usuario:")
    num_recommendations = st.number_input(
        "Número de recomendaciones:", min_value=1, value=10, step=1
    )

    if st.button("Obtener recomendaciones"):
        if id_input:
            with st.spinner("Por favor espere..."):
                try:
                    customer_id = int(id_input)
                    df = spark.createDataFrame([(customer_id,)], ["customer_id"])
                    recommendations = model.recommendForUserSubset(
                        df, numItems=int(num_recommendations)
                    )
                    if recommendations.count() == 0:
                        st.warning(
                            "No se encontraron recomendaciones para este usuario."
                        )
                    else:
                        recommendations_for_user = recommendations.select(
                            "recommendations"
                        ).collect()
                        st.subheader(f"Recomendaciones para el usuario {customer_id}:")
                        recommendation_items = recommendations_for_user[
                            0
                        ].recommendations
                        predictions_df = spark.createDataFrame(recommendation_items)
                        predictions_df = inverter.transform(predictions_df).toPandas()
                        predictions_df.rename(
                            columns={"original_item_id": "product_id"}, inplace=True
                        )
                        predictions_df.drop("item_id", axis=1, inplace=True)
                        # Puts item_id as the first column
                        cols = list(predictions_df.columns)
                        cols = [cols[-1]] + cols[:-1]
                        predictions_df = predictions_df[cols]
                        st.dataframe(predictions_df)
                except ValueError:
                    st.error("Por favor introduce un ID válido.")
        else:
            st.warning("Por favor introduce un ID.")


if __name__ == "__main__":
    main()
