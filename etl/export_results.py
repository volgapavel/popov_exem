"""
etl/export_results.py
=====================
Экспорт артефактов (модель + метрики).

Режимы работы
-------------
* local  – просто гарантирует, что обе цели лежат в results/ (дефолт).
* s3     – загружает в указанный S3-бакет.  Авторизация идёт через
           переменные окружения AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
           (или профили в ~/.aws/credentials).

Пример CLI
----------
$ python etl/export_results.py                      # локально
$ python etl/export_results.py --mode s3           \
      --bucket ml-artifacts --prefix bc_demo/
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
MODEL = RESULTS_DIR / "model.pkl"
METRICS = RESULTS_DIR / "metrics.json"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --------------------------------------------------------------------- #
def _export_local(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in (MODEL, METRICS):
        shutil.copy2(f, out_dir / f.name)
        logger.info("Copied %s → %s", f, out_dir / f.name)


def _export_s3(bucket: str, prefix: str = ""):
    import boto3
    s3 = boto3.client("s3")  # cred-ы читаются автоматом из env/ ~/.aws
    for f in (MODEL, METRICS):
        key = f"{prefix}{f.name}"
        s3.upload_file(str(f), bucket, key)
        logger.info("Uploaded %s to s3://%s/%s", f, bucket, key)


def export_results(mode: str = "local",
                   bucket: str | None = None,
                   prefix: str = "",
                   out_dir: str | Path = RESULTS_DIR / "export"):
    if mode == "local":
        _export_local(Path(out_dir))
    elif mode == "s3":
        if not bucket:
            raise ValueError("--bucket обязателен для mode=s3")
        _export_s3(bucket, prefix)
    else:
        raise ValueError("mode must be 'local' or 's3'")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local", "s3"], default="local")
    p.add_argument("--bucket", help="S3 bucket name (для mode=s3)")
    p.add_argument("--prefix", default="", help="S3 key prefix")
    p.add_argument("--out_dir", default=RESULTS_DIR / "export",
                   help="Локальный каталог для copy, если mode=local")
    args = p.parse_args()
    export_results(args.mode, args.bucket, args.prefix, args.out_dir)
