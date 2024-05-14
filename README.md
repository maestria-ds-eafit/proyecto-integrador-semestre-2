# Proyecto integrador

Proyecto integrador para el segundo semestre de la maestría de ciencia de datos y analítica de EAFIT.

## Configuración para usar AWS EMR

* Instalar las dependencias de Python: `pdm install`

* Instalar el AWS CLI: <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>

* Primero configurar el CLI de AWS.

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

* `pipenv shell`
* `spark-submit <script_name>.py`

Se recomienda que se ejecute solo con un sample de los datos, para no sobrecargar el sistema.

## Para descargar los modelos del bucket

* `aws s3 cp --recursive s3://amazon-reviews-eafit/model-random-stratified-split-sample model-random-stratified-split-sample`
* `aws s3 cp --recursive s3://amazon-reviews-eafit/inverter-random-stratified-split-sample inverter-random-stratified-split-sample`
