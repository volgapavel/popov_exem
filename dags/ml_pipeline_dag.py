"""
ml_pipeline_dag.py
==================
Airflow orchestration for the Breast-Cancer diagnostic pipeline.
Four linear tasks, one per ETL stage.

Run cadence  : manual by default (set schedule_interval='@daily' to run daily)
Author       : P. Popov
Created      : 2025-06-17
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from etl.export_results import export_results
# import the callable steps from etl package
from etl.load_data import load_data
from etl.preprocess_data import preprocess_data
from etl.train_model import train_model
from etl.evaluate_metrics import evaluate_metrics

DAG_ID = "ml_pipeline_breast_cancer"

default_args = {
    "owner": "data-team",
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    description="End-to-end ML ETL for Breast-Cancer diagnostic dataset",
    start_date=datetime(2025, 6, 1),
    schedule_interval=None,           # change to '@daily' for daily run
    catchup=False,                    # do not back-fill missed periods
    default_args=default_args,
    tags=["ml", "breast-cancer", "logreg"],
) as dag:

    t1 = PythonOperator(
        task_id="load_data",
        python_callable=load_data,     # uses defaults → writes results/data_raw.csv
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t4 = PythonOperator(
        task_id="evaluate_metrics",
        python_callable=evaluate_metrics,
    )

    t5 = PythonOperator(
        task_id="export_results",
        python_callable=export_results,  # локально
        # op_kwargs={"mode": "s3", "bucket": "ml-artifacts", "prefix": "bc_demo/"},
    )

    # linear dependency chain
    t1 >> t2 >> t3 >> t4 >> t5
