import os

import awswrangler as wr
import streamlit as st
from dotenv import load_dotenv
from pyspark.ml.feature import IndexToString
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

load_dotenv()

spark = SparkSession.builder.appName("Recommendations App").getOrCreate()  # type: ignore
bucket_name = os.getenv("BUCKET_NAME")
model = ALSModel.load("model-random-stratified-split-sample")
inverter = IndexToString.load("inverter-random-stratified-split-sample")

data_training = wr.s3.read_parquet(path="s3://amazon-reviews-eafit/sample-training/")
data_test = wr.s3.read_parquet(path="s3://amazon-reviews-eafit/sample-test/")


def main():
    st.title("Sistema de recomendación para Amazon")

    id_input = st.text_input("Introduce el ID del usuario:")
    num_recommendations = st.number_input(
        "Número de recomendaciones:", min_value=1, value=10, step=1
    )

    if st.button("Obtener recomendaciones"):
        if id_input:
            customer_id = int(id_input)
            products_bought_training = data_training[
                data_training["customer_id"] == customer_id
            ][["product_id", "product_title", "product_category"]]
            products_bought_test = data_test[data_test["customer_id"] == customer_id][
                ["product_id", "product_title", "product_category"]
            ]
            st.write("Productos comprados por el usuario (en el training set):")
            st.dataframe(products_bought_training, hide_index=True)
            st.write("Productos comprados por el usuario (en el test set):")
            st.dataframe(products_bought_test, hide_index=True)
            with st.spinner("Por favor espere..."):
                try:
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
                        product_ids = predictions_df["product_id"].tolist()
                        products_df = data_training[
                            data_training["product_id"].isin(product_ids)
                        ]
                        predictions_df = predictions_df.merge(
                            products_df[
                                ["product_id", "product_title", "product_category"]
                            ],
                            on="product_id",
                        )
                        predictions_df.drop("item_id", axis=1, inplace=True)
                        predictions_df = predictions_df[
                            [
                                "product_id",
                                "product_title",
                                "product_category",
                                "rating",
                            ]
                        ]
                        st.dataframe(predictions_df, hide_index=True)
                except ValueError:
                    st.error("Por favor introduce un ID válido.")
        else:
            st.warning("Por favor introduce un ID.")


if __name__ == "__main__":
    main()
