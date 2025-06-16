"""
etl/train_model.py
==================
Шаг 3 ETL-конвейера: обучение модели LogisticRegression.

Алгоритм
--------
1. Читаем «чистый» датасет (results/data_clean.csv).
2. Делим на train / test (80 % / 20 %, random_state=42, стратификация по `diagnosis`).
3. Кодируем метки: Benign → 0, Malignant → 1.
4. Обучаем LogisticRegression (solver='liblinear', max_iter=1000).
5. Считаем Accuracy на тесте – логируем для контроля.
6. Сохраняем:
   • модель `results/model.pkl` (joblib.dump, compressed=True);
   • тестовый CSV `results/test_data.csv`
     (30 признаков + diagnosis — **без** предсказаний, чтобы последующий
      шаг evaluate_metrics сам их получал).

Запуск
------
$ python etl/train_model.py
$ python etl/train_model.py --clean_csv results/data_clean.csv --out_dir results
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
DEFAULT_CLEAN_CSV = Path(os.getenv("CLEAN_CSV", "../results/data_clean.csv"))
DEFAULT_OUT_DIR = Path(os.getenv("OUT_DIR", "results"))
MODEL_FILENAME = "model.pkl"
TEST_CSV_NAME = "test_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20 %


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def _load_dataset(clean_csv: Path | str) -> Tuple[pd.DataFrame, pd.Series]:
    """Загружает датасет и возвращает X, y (diagnosis)."""
    df = pd.read_csv(clean_csv)
    if "diagnosis" not in df.columns:
        raise ValueError("В clean CSV отсутствует колонка 'diagnosis'.")

    X = df.drop(columns="diagnosis")
    y = df["diagnosis"].map({"B": 0, "M": 1})  # бинаризация
    return X, y


# --------------------------------------------------------------------------- #
# Основная функция
# --------------------------------------------------------------------------- #
def train_model(clean_csv: Path | str = DEFAULT_CLEAN_CSV,
                out_dir: Path | str = DEFAULT_OUT_DIR) -> str:
    """
    Обучает LogisticRegression, сохраняет модель и тестовый набор.

    Параметры
    ---------
    clean_csv : str | Path
        Путь к подготовленному датасету (`data_clean.csv`).
    out_dir : str | Path
        Папка для сохранения `model.pkl` и `test_data.csv`.

    Возврат
    -------
    str – абсолютный путь к сохранённой модели.
    """
    clean_csv = Path(clean_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clean_csv.exists():
        raise FileNotFoundError(f"Clean CSV not found: {clean_csv}")

    # 1. Загружаем данные
    X, y = _load_dataset(clean_csv)

    # 2. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train/test split: train=%d, test=%d", len(X_train), len(X_test))

    # 3. Обучаем модель
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    logger.info("Модель обучена: %s", model)

    # 4. Базовая оценка Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Accuracy на тесте: %.4f", acc)

    # 5. Сохраняем модель
    model_path = out_dir / MODEL_FILENAME
    joblib.dump(model, model_path, compress=True)
    logger.info("Model saved: %s", model_path.resolve())

    # 6. Сохраняем тестовый CSV (признаки + diagnosis в строках)
    #    Восстанавливаем строки diagnosis B/M для удобства следующих шагов
    diagnosis_series = y_test.map({0: "B", 1: "M"}).rename("diagnosis")
    test_df = pd.concat([X_test.reset_index(drop=True), diagnosis_series.reset_index(drop=True)], axis=1)
    test_csv_path = out_dir / TEST_CSV_NAME
    test_df.to_csv(test_csv_path, index=False)
    logger.info("Test data saved: %s", test_csv_path.resolve())

    return str(model_path.resolve())


# --------------------------------------------------------------------------- #
# CLI-обёртка
# --------------------------------------------------------------------------- #
def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение LogisticRegression на Breast Cancer")
    parser.add_argument("--clean_csv", default=DEFAULT_CLEAN_CSV,
                        help="Path to data_clean.csv (default: results/data_clean.csv)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="Directory to save model & test data (default: results)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    train_model(clean_csv=args.clean_csv, out_dir=args.out_dir)
