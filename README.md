# Proyecto integrador

Proyecto integrador para el segundo semestre de la maestría de ciencia de datos y analítica de EAFIT.

## Prerrequisitos

* Tener `pdm` instalado (Python Dependency Manager): <https://pdm-project.org/en/latest/>

* Tener una cuenta de AWS y al `aws cli` instalado: <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>

## Estructura del proyecto

* El proyecto utiliza **PDM (Python Dependency Manager)** para el manejo de dependencias. Esto implica la existencia de un archivo **pyproject.toml** y **pdm.lock**. Cualquier versión de Python 3.11 sirve para este proyecto.
* En el root del proyecto estos son los archivos importantes:
  * `.env.template`: Este archivo debe ser copiado a un archivo .env (que está en el .gitignore) para que los programas lean estas variables de entorno
  * `main.py`: archivo utilitario utilizado para mandar Jobs a AWS EMR.
  * `app.py`: archivo para correr el demo de Streamlit
* En la carpeta *scripts* se puede encontrar los siguientes archivos:
  * `data_cleaning.py`: script para la limpieza de la data y para reporte de conteos por cada paso de limpieza
  * `eda.py`: script para sacar agregados de la data refinada
  * get_reviews_count_distribution.py: script para calcular la distribución de reviews. Es decir, cuántos usuarios dieron “x” cantidad de reviews.
  * `get_sample_for_demo.py`: script para obtener un sampleo de la data para la demo con Streamlit
  * `get_sample_for_model.py`: script para obtener un sampleo de la data para el grid search
  * `recommender_random_split.py`: script para la implementación de un sistema de recomendación utilizando el split de data randomizado.
  * `recommender_random_stratified.py`: script para el sistema de recomendación utilizando el split de data estratificado y randomizado.
  * `recommender_stratified_split.py`: script para el sistema de recomendación usando el split estratificado.
  * `total_rows_per_category.py`: script para sacar el total de registros por categoría
  * `validation.py`: script para hacer grid search con un porcentaje de la data

## Configuración para usar AWS EMR

* Instalar las dependencias de Python: `pdm install`

* Configurar el CLI de AWS.

```terminal
aws configure
```

* Primero hay que crear un IAM Role con permisos para EMR y S3.

```terminal
aws iam create-role \
    --role-name EMRServerlessS3RuntimeRole \
    --assume-role-policy-document file://emr-serverless-trust-policy.json
```

* Segundo, crear un IAM policy para la carga de trabajo. Este policy provee acceso de lectura a archivos guardados en S3 y acceso de lectura y escritura a archivos guardados en el bucket `amazon-reviews-eafit`.

```terminal
aws iam create-policy \
    --policy-name EMRServerlessS3AndGlueAccessPolicy \
    --policy-document file://emr-sample-access-policy.json
```

* Tercero, adjuntar el IAM policy `EMRServerlessS3AndGlueAccessPolicy` al job runtime role `EMRServerlessS3RuntimeRole`:

```terminal
aws iam attach-role-policy \
    --role-name EMRServerlessS3RuntimeRole \
    --policy-arn <policy-arn>
```

Donde `policy-arn` es el ARN del policy, es decir el ARN de `EMRServerlessS3AndGlueAccessPolicy`.

* Cuarto, crear una aplicación ser EMR Serverless:

```terminal
aws emr-serverless create-application \
    --release-label emr-7.0.0 \
    --type "SPARK" \
    --name proyecto-integrador-semestre-2
```

* Quinto, existe un script llamado `main.py` que permite ejecutar jobs en EMR Serverless. Para ejecutar un job, se debe ejecutar el siguiente comando:

```terminal
python main.py scripts/<nombre_del_script>.py
```

> IMPORTANTE: llenar las variables de entorno en el archivo `.env` con la información del `.env.template`.

## Ejecución de scripts local

Para ejecutar un script local:

* `spark-submit <script_name>.py`

Se recomienda que se ejecute solo con un sample de los datos, para no sobrecargar el sistema.

## Para descargar los modelos del bucket

* `aws s3 cp --recursive s3://amazon-reviews-eafit/model-random-stratified-split-sample model-random-stratified-split-sample`
* `aws s3 cp --recursive s3://amazon-reviews-eafit/inverter-random-stratified-split-sample inverter-random-stratified-split-sample`
