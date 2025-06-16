"""
etl/load_data.py
================
Шаг 1 ETL-конвейера: загрузка и первичный анализ Breast Cancer
Wisconsin Diagnostic dataset.

• Может работать двумя способами:
  1) Cкачивает готовый набор через `sklearn.datasets.load_breast_cancer()`
     – интернет не требуется, данные поставляются вместе со scikit-learn.
  2) Берёт сырой CSV-файл `wdbc.data`, скачанный с UCI-репозитория
     (путь передаётся через `--use_local_csv`).

• Выполняет мини-EDA: число строк/столбцов, распределение классов.

• Сохраняет неизменённый датасет в `results/data_raw.csv`
  (каталог задаётся `--out_dir` или переменной окружения OUT_DIR).

Можно запускать:
  $ python etl/load_data.py                       # sklearn-вариант
  $ python etl/load_data.py --use_local_csv ./wdbc.data

Из Airflow DAG файл импортируется как модуль и вызывается
функцией `load_data()`.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.datasets import load_breast_cancer

# --------------------------------------------------------------------------- #
# Логирование
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #
DEFAULT_OUT_DIR = Path(os.getenv("OUT_DIR", "results"))
RAW_CSV_NAME = "data_raw.csv"


# --------------------------------------------------------------------------- #
# Функции загрузки
# --------------------------------------------------------------------------- #
def _load_from_sklearn() -> pd.DataFrame:
    """Берёт датасет через scikit-learn, конвертирует в DataFrame."""
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame
    # Приводим целевой столбец к привычным меткам B/M
    df["diagnosis"] = df["target"].map({0: "B", 1: "M"})
    df.drop(columns="target", inplace=True)
    return df


def _load_from_csv(csv_path: Path) -> pd.DataFrame:
    """Читает сырой UCI-файл wdbc.data и назначает названия колонок."""
    df = pd.read_csv(csv_path, header=None)
    # Если заголовков нет, вешаем правильные имена (32 колонки)
    if list(df.columns) == list(range(32)):
        id_col = ["id"]
        diag_col = ["diagnosis"]
        features = [
            # 30 признаков (mean, se, worst) – полный список из описания набора
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave_points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave_points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave_points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ]
        df.columns = id_col + diag_col + features
    return df


# --------------------------------------------------------------------------- #
# Основная функция
# --------------------------------------------------------------------------- #
def load_data(out_dir: Path | str = DEFAULT_OUT_DIR,
              source_csv: Optional[str | Path] = None) -> str:
    """
    Загружает датасет, логирует базовую статистику и сохраняет CSV.

    Параметры
    ---------
    out_dir : Path | str
        Каталог, куда записывать `data_raw.csv`.
    source_csv : str | Path | None
        Путь к локальному wdbc.data. Если None – используется sklearn-версия.

    Возврат
    -------
    str – абсолютный путь к сохранённому CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / RAW_CSV_NAME

    if source_csv:
        logger.info("Загружаю датасет из локального файла: %s", source_csv)
        df = _load_from_csv(Path(source_csv))
    else:
        logger.info("Загружаю датасет через sklearn.datasets.load_breast_cancer()")
        df = _load_from_sklearn()

    # Мини-EDA
    n_rows, n_cols = df.shape
    logger.info("Размер датасета: %d объектов, %d столбцов", n_rows, n_cols)

    if "diagnosis" in df.columns:
        cls_cnt = df["diagnosis"].value_counts().to_dict()
        logger.info("Распределение классов: %s",
                    ", ".join(f"{k}={v}" for k, v in cls_cnt.items()))

    df.to_csv(out_csv, index=False)
    logger.info("Сырой CSV сохранён: %s", out_csv.resolve())

    return str(out_csv.resolve())


# --------------------------------------------------------------------------- #
# CLI-обёртка
# --------------------------------------------------------------------------- #
def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Загрузка Breast Cancer данных")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="Папка для сохранения CSV (по умолчанию 'results').")
    parser.add_argument("--use_local_csv", metavar="PATH",
                        help="Путь к wdbc.data; если не указан, берётся вариант из sklearn.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    load_data(out_dir=args.out_dir, source_csv=args.use_local_csv)
