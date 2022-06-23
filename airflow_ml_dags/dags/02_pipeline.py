from datetime import timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable


mount_dir = Variable.get("MOUNT_DIR")

default_args = {
    "owner": "airflow",
    "email": ["chugaev.r@mail.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="02_pipeline",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(7),
) as dag:

    wait_raw_data = FileSensor(
        task_id="wait_raw_data",
        filepath="raw/{{ ds }}/features.csv",
        fs_conn_id="fs_conn",
        poke_interval=30,
        retries=100,
    )

    wait_raw_target = FileSensor(
        task_id="wait_raw_target",
        filepath="raw/{{ ds }}/targets.csv",
        fs_conn_id="fs_conn",
        poke_interval=30,
        retries=100,
    )

    splitter = DockerOperator(
        image="airflow-splitter",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/splitted/{{ ds }}",
        mount_tmp_dir=False,
        network_mode="bridge",
        mounts=[
            Mount(
                source=mount_dir,
                target="/data",
                type="bind",
            )
        ],
        task_id="airflow-splitter",
    )

    preprocessor = DockerOperator(
        image="airflow-preprocessor",
        command="--input_dir /data/splitted/{{ ds }} "
                "--output_dir /data/processed/{{ ds }}",
        mount_tmp_dir=False,
        network_mode="bridge",
        mounts=[
            Mount(
                source=mount_dir,
                target="/data",
                type="bind",
            )
        ],
        task_id="airflow-preprocessor",
    )

    trainer = DockerOperator(
        image="airflow-trainer",
        command="--features_dir /data/processed/{{ ds }} "
                "--targets_dir /data/splitted/{{ ds }} "
                "--output_dir /data/models/{{ ds }}",
        mount_tmp_dir=False,
        network_mode="bridge",
        mounts=[
            Mount(
                source=mount_dir,
                target="/data",
                type="bind",
            )
        ],
        task_id="airflow-trainer",
    )

    validator = DockerOperator(
        image="airflow-validator",
        command="--model_path /data/models/{{ ds }}/model.pkl "
                "--features_dir /data/processed/{{ ds }} "
                "--targets_dir /data/splitted/{{ ds }} "
                "--output_dir /data/metrics/{{ ds }}",
        mount_tmp_dir=False,
        network_mode="bridge",
        mounts=[
            Mount(
                source=mount_dir,
                target="/data",
                type="bind",
            )
        ],
        task_id="airflow-validator",
    )

    [wait_raw_data, wait_raw_target] >> splitter >> preprocessor >> trainer >> validator
