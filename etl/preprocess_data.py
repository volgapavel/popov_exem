"""
etl/preprocess_data.py
======================
Шаг 2 ETL-конвейера: очистка и предобработка датасета Breast Cancer Wisconsin
Diagnostic.

Функциональность
----------------
1. Загружает CSV с «сырыми» данными (`results/data_raw.csv` по умолчанию).
2. Удаляет неинформативную колонку `id`, если она присутствует.
3. Унифицирует заголовки признаков — заменяет пробелы на `_`, приводит к lower-case.
4. Проверяет целостность:
   • наличие колонки `diagnosis`;
   • отсутствие пропусков в 30 числовых признаках.
5. Масштабирует признаки `StandardScaler`-ом (z-score).
6. Сохраняет «чистый» датасет в `results/data_clean.csv`.
7. Дополнительно сериализует обученный `scaler` в `results/scaler.pkl`
   (можно использовать на инференсе).

Запуск из CLI
--------------
$ python etl/preprocess_data.py
$ python etl/preprocess_data.py --raw_csv other.csv --out_dir another_folder

Использование из Airflow DAG
----------------------------
from etl.preprocess_data import preprocess_data

t2 = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,          # без аргументов → дефолтные пути
)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Логирование
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)

# --------------------------------------------------------------------------- #
# Константы и вспомогательные функции
# --------------------------------------------------------------------------- #
DEFAULT_RAW_CSV = Path(os.getenv("RAW_CSV", "../results/data_raw.csv"))
DEFAULT_OUT_DIR = Path(os.getenv("OUT_DIR", "results"))
CLEAN_CSV_NAME = "data_clean.csv"
SCALER_FILENAME = "scaler.pkl"


def _standardize_headers(columns: Sequence[str]) -> list[str]:
    """
    Приводим названия к snake_case: пробелы → '_', нижний регистр.
    """
    return [col.strip().replace(" ", "_").lower() for col in columns]


# --------------------------------------------------------------------------- #
# Основная функция
# --------------------------------------------------------------------------- #
def preprocess_data(raw_csv: Path | str = DEFAULT_RAW_CSV,
                    out_dir: Path | str = DEFAULT_OUT_DIR) -> str:
    """
    Выполняет очистку и масштабирование признаков.

    Параметры
    ---------
    raw_csv : str | Path
        Путь к CSV с сырым набором (результат шага «load_data»).
    out_dir : str | Path
        Папка для сохранения «чистого» набора и scaler-а.

    Возврат
    -------
    str — абсолютный путь к `data_clean.csv`.
    """
    raw_csv = Path(raw_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_csv}")

    logger.info("Читаю сырые данные: %s", raw_csv)
    df = pd.read_csv(raw_csv)

    # 1. Удаляем неинформативный ID, если есть
    if "id" in df.columns:
        df = df.drop(columns="id")
        logger.debug("Колонка 'id' удалена.")

    # 2. Унификация заголовков
    df.columns = _standardize_headers(df.columns)

    # 3. Проверки целостности
    if "diagnosis" not in df.columns:
        raise ValueError("Ожидаемая колонка 'diagnosis' отсутствует.")

    feature_cols = [c for c in df.columns if c != "diagnosis"]
    if len(feature_cols) != 30:
        logger.warning("Ожидается 30 числовых признаков, получено: %d", len(feature_cols))

    if df[feature_cols].isnull().any().any():
        n_missing = df[feature_cols].isnull().sum().sum()
        raise ValueError(f"Обнаружено {n_missing} пропущенных значений в признаках.")

    # 4. Масштабирование признаков (z-score)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    logger.info("Стандартизация завершена (μ≈0, σ≈1).")

    # 5. Сохранение
    clean_csv = out_dir / CLEAN_CSV_NAME
    df.to_csv(clean_csv, index=False)
    logger.info("Очищенный датасет сохранён: %s", clean_csv.resolve())

    scaler_path = out_dir / SCALER_FILENAME
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler сериализован: %s", scaler_path.resolve())

    return str(clean_csv.resolve())


# --------------------------------------------------------------------------- #
# CLI-обёртка
# --------------------------------------------------------------------------- #
def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Очистка и предобработка данных Breast Cancer")
    parser.add_argument("--raw_csv", default=DEFAULT_RAW_CSV,
                        help="Путь к data_raw.csv (по умолчанию results/data_raw.csv)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="Каталог для сохранения результатов (по умолчанию 'results').")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    preprocess_data(raw_csv=args.raw_csv, out_dir=args.out_dir)
