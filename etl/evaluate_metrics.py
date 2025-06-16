"""
etl/evaluate_metrics.py
=======================
Шаг 4 ETL-конвейера: оценка качества модели LogisticRegression
на отложенной тестовой выборке.

1. Загружает:
   • обученную модель (`results/model.pkl`);
   • CSV с тестовой выборкой (`results/test_data.csv`) — содержит
     все 30 признаков + колонку `diagnosis` (B/M).

2. Делит DataFrame на X (признаки) и y (метки), получает прогнозы.

3. Вычисляет Accuracy, Precision, Recall, F1-score (класс M — «злокач.»
   принимается положительным).

4. Сохраняет метрики в JSON-файл `results/metrics.json`
   и выводит значения в лог.

Запуск
------
$ python etl/evaluate_metrics.py
$ python etl/evaluate_metrics.py --model results/model.pkl \
                                 --test_csv results/test_data.csv \
                                 --out_dir results
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

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
DEFAULT_MODEL = Path(os.getenv("MODEL_PKL", "../results/model.pkl"))
DEFAULT_TEST_CSV = Path(os.getenv("TEST_CSV", "../results/test_data.csv"))
DEFAULT_OUT_DIR = Path(os.getenv("OUT_DIR", "results"))
METRICS_FILENAME = "metrics.json"


# --------------------------------------------------------------------------- #
# Основная функция
# --------------------------------------------------------------------------- #
def evaluate_metrics(model_path: Path | str = DEFAULT_MODEL,
                     test_csv: Path | str = DEFAULT_TEST_CSV,
                     out_dir: Path | str = DEFAULT_OUT_DIR) -> str:
    """
    Вычисляет Accuracy, Precision, Recall, F1 и сохраняет их в JSON.

    Параметры
    ---------
    model_path : str | Path
        Файл с сериализованной моделью (pickle/joblib).
    test_csv : str | Path
        CSV с тестовой выборкой (30 признаков + 'diagnosis').
    out_dir : str | Path
        Папка, куда сохранить metrics.json.

    Возврат
    -------
    str – абсолютный путь к сохранённому JSON-файлу с метриками.
    """
    model_path = Path(model_path)
    test_csv = Path(test_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    logger.info("Загружаю модель: %s", model_path)
    model = joblib.load(model_path)

    logger.info("Читаю тестовые данные: %s", test_csv)
    df_test = pd.read_csv(test_csv)

    if "diagnosis" not in df_test.columns:
        raise ValueError("В тестовом CSV отсутствует колонка 'diagnosis'.")

    X_test = df_test.drop(columns="diagnosis")
    y_test = df_test["diagnosis"].map({"B": 0, "M": 1})  # бинарные метки 0/1

    # Предсказания
    y_pred = model.predict(X_test)

    # Метрики (класс '1' == 'M' — положительный)
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Доп. отчёт в лог
    logger.info("=== Classification Report ===\n%s",
                classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
    logger.info("=== Сводные метрики ===")
    for k, v in metrics.items():
        logger.info("  %-10s: %.4f", k, v)

    # Сохраняем JSON
    metrics_path = out_dir / METRICS_FILENAME
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    logger.info("Метрики сохранены: %s", metrics_path.resolve())
    return str(metrics_path.resolve())


# --------------------------------------------------------------------------- #
# CLI-обёртка
# --------------------------------------------------------------------------- #
def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка метрик модели Breast Cancer")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to model.pkl (default: results/model.pkl)")
    parser.add_argument("--test_csv", default=DEFAULT_TEST_CSV,
                        help="Path to test_data.csv (default: results/test_data.csv)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="Directory to save metrics.json (default: results)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    evaluate_metrics(model_path=args.model, test_csv=args.test_csv, out_dir=args.out_dir)
