# Proyecto integrador

Proyecto integrador para el segundo semestre de la maestría de ciencia de datos y analítica de EAFIT.

## Configuración para usar AWS EMR

* Instalar las dependencias de Python: `pipenv install`

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

* Quinto, crear un job de EMR Serverless:

```terminal
aws emr-serverless start-job-run \
    --application-id <application-id> \
    --execution-role-arn <job-role-arn> \
    --name job-run-name \
    --job-driver '{
        "sparkSubmit": {
          "entryPoint": "s3://amazon-reviews-eafit/scripts/recommender.py",
          "sparkSubmitParameters": "--conf spark.executor.cores=1 --conf spark.executor.memory=4g --conf spark.driver.cores=1 --conf spark.driver.memory=4g --conf spark.executor.instances=1 --conf spark.kryoserializer.buffer.max=512m"
        }
    }'
```

`application-id` es el ID de la aplicación creada en el paso anterior.
`job-role-arn` es el ARN de `EMRServerlessS3RuntimeRole`

## Ejecución de scripts local

Para ejecutar un script local:

* `pipenv shell`
* `spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.4 <script_name>.py`
