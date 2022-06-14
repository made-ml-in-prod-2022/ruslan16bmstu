from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.utils.dates import days_ago


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=60),
}

with DAG(
    dag_id="01_generator",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
) as dag:
    generator = DockerOperator(
        image="airflow-generator",
        command="/data/raw/{{ ds }}",
        mount_tmp_dir=False,
        network_mode="bridge",
        mounts=[
            Mount(
                source="/home/ruslan16/ml_in_prod/ruslan16bmstu/airflow_ml_dags/data",
                target="/data",
                type="bind",
            )
        ],
        task_id="airflow-generator",
    )
