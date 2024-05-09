import os
import sys

import boto3
from dotenv import load_dotenv

load_dotenv()


def upload_to_s3(file_path, bucket_name, folder_name):
    """
    Uploads a file to an S3 bucket in a specific folder.
    """
    s3 = boto3.client("s3")
    file_name = os.path.basename(file_path)
    s3_key = folder_name + "/" + file_name
    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"File uploaded to S3 bucket: {bucket_name}/{s3_key}")
        return s3_key
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return None


def run_emr_job(s3_key):
    """
    Runs an EMR job using AWS CLI.
    """

    application_id = os.getenv("APPLICATION_ID")
    execution_role_arn = os.getenv("EXECUTION_ROLE_ARN")
    spark_executor_cores = os.getenv("SPARK_EXECUTOR_CORES", default="4")
    spark_executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY", default="16g")
    spark_driver_cores = os.getenv("SPARK_DRIVER_CORES", default="1")
    spark_driver_memory = os.getenv("SPARK_DRIVER_MEMORY", default="4g")
    spark_executor_instances = os.getenv("SPARK_EXECUTOR_INSTANCES", default="1")
    spark_kryoserializer_buffer_max = os.getenv(
        "SPARK_KRYOSERIALIZER_BUFFER_MAX", default="512m"
    )

    command = [
        "aws",
        "emr-serverless",
        "start-job-run",
        "--application-id",
        application_id,
        "--execution-role-arn",
        execution_role_arn,
        "--name",
        s3_key,
        "--job-driver",
        f'\'{{"sparkSubmit": {{"entryPoint": "s3://{bucket_name}/{file_path}", "sparkSubmitParameters": "--conf spark.kryoserializer.buffer.max={spark_kryoserializer_buffer_max} --conf spark.executor.instances={spark_executor_instances} --conf spark.driver.memory={spark_driver_memory} --conf spark.driver.cores={spark_driver_cores} --conf spark.executor.memory={spark_executor_memory} --conf spark.executor.cores={spark_executor_cores}"}}}}\'',
    ]
    try:
        string_command = " ".join(command)
        os.system(string_command)
        print("EMR job started successfully.")
    except Exception as e:
        print(f"Error running EMR job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    bucket_name = os.getenv("BUCKET_NAME")
    folder_name = "scripts"

    try:
        s3_key = upload_to_s3(file_path, bucket_name, folder_name)
    except Exception as e:
        print(f"Error while uploading to S3: {e}")
        sys.exit(1)

    # Run EMR job
    run_emr_job(s3_key)
