# Proyecto integrador

Proyecto integrador para el segundo semestre de la maestría de ciencia de datos y analítica de EAFIT.

## Configuración para usar AWS EMR

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
    --policy-arn policy-arn
```

Donde `policy-arn` es el ARN del trust policy, es decir el ARN de `EMRServerlessS3RuntimeRole`.

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
    --application-id application-id \
    --execution-role-arn job-role-arn \
    --name job-run-name \
    --job-driver '{
        "sparkSubmit": {
          "entryPoint": "s3://DOC-EXAMPLE-BUCKET/scripts/wordcount.py",
          "entryPointArguments": ["s3://DOC-EXAMPLE-BUCKET/emr-serverless-spark/output"],
          "sparkSubmitParameters": "--conf spark.executor.cores=1 --conf spark.executor.memory=4g --conf spark.driver.cores=1 --conf spark.driver.memory=4g --conf spark.executor.instances=1"
        }
    }'
```
