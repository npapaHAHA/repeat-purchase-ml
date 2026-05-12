import argparse
import json
import sqlite3
from contextlib import closing
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES


HORIZON = pd.Timedelta(days=30)
LOOKBACK = pd.Timedelta(days=90)
ORDER_COLUMNS = ["Invoice", "Quantity", "InvoiceDate", "Price", "Customer ID"]

SNAPSHOT_QUERY = """
WITH history AS (
    SELECT customer_id,
           COUNT(*) AS order_count,
           SUM(order_value) AS total_spend,
           AVG(order_value) AS mean_order_value,
           JULIANDAY(:cutoff) - JULIANDAY(MAX(ordered_at)) AS days_since_last_order
    FROM orders
    WHERE ordered_at >= :history_start AND ordered_at < :cutoff
    GROUP BY customer_id
), future_buyers AS (
    SELECT DISTINCT customer_id
    FROM orders
    WHERE ordered_at >= :cutoff AND ordered_at < :outcome_end
)
SELECT history.*,
       CASE WHEN future_buyers.customer_id IS NULL THEN 0 ELSE 1 END AS target
FROM history
LEFT JOIN future_buyers USING (customer_id)
ORDER BY customer_id
"""


def load_orders(data_dir):
    cache = data_dir / "retail-orders.csv"
    if cache.is_file():
        return pd.read_csv(cache, dtype={"ordered_at": str})

    with ZipFile(data_dir / "retail.zip") as archive:
        workbook = next(name for name in archive.namelist() if name.endswith(".xlsx"))
        with BytesIO(archive.read(workbook)) as source:
            sheets = pd.read_excel(source, sheet_name=None, usecols=ORDER_COLUMNS)
    rows = pd.concat(sheets.values(), ignore_index=True).dropna(subset=ORDER_COLUMNS)
    valid = (
        rows["Quantity"].gt(0)
        & rows["Price"].gt(0)
        & ~rows["Invoice"].astype(str).str.startswith(("C", "c"))
    )
    rows = rows.loc[valid].copy()
    rows["customer_id"] = rows["Customer ID"].astype(int)
    rows["ordered_at"] = pd.to_datetime(rows["InvoiceDate"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    rows["line_value"] = rows["Quantity"] * rows["Price"]
    orders = rows.groupby(["customer_id", "Invoice"], as_index=False).agg(
        ordered_at=("ordered_at", "min"),
        order_value=("line_value", "sum"),
    )[["customer_id", "ordered_at", "order_value"]]
    partial = cache.with_suffix(".tmp")
    orders.to_csv(partial, index=False)
    partial.replace(cache)
    return orders


def build_snapshots(orders):
    first_date = pd.Timestamp(orders["ordered_at"].min()).ceil("D")
    last_date = pd.Timestamp(orders["ordered_at"].max()).floor("D")
    dates = pd.date_range(first_date + LOOKBACK, last_date - HORIZON, freq="30D")
    if len(dates) < 5:
        raise ValueError("Expected at least five complete prediction windows")

    snapshots = []
    with closing(sqlite3.connect(":memory:")) as connection:
        orders.to_sql("orders", connection, index=False)
        connection.execute("CREATE INDEX orders_by_date ON orders(ordered_at)")
        for cutoff in dates:
            parameters = {
                "history_start": str(cutoff - LOOKBACK),
                "cutoff": str(cutoff),
                "outcome_end": str(cutoff + HORIZON),
            }
            snapshot = pd.read_sql_query(SNAPSHOT_QUERY, connection, params=parameters)
            snapshots.append(snapshot.assign(cutoff=cutoff))
    return dates, pd.concat(snapshots, ignore_index=True)


def evaluate(rows, scores):
    return {
        "average_precision": float(average_precision_score(rows["target"], scores)),
        "roc_auc": float(roc_auc_score(rows["target"], scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dates, rows = build_snapshots(load_orders(args.data_dir))
    train = rows[rows["cutoff"] < dates[-4]]
    validation = rows[rows["cutoff"].isin(dates[-4:-2])]
    test = rows[rows["cutoff"].isin(dates[-2:])]
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "catboost": CatBoostClassifier(
            iterations=400, depth=5, learning_rate=0.05, random_seed=42,
            verbose=False, allow_writing_files=False, thread_count=4,
        ),
    }
    validation_metrics = {"constant_baseline": evaluate(validation, [0] * len(validation))}
    for name, model in models.items():
        model.fit(train[FEATURES], train["target"])
        validation_metrics[name] = evaluate(validation, model.predict_proba(validation[FEATURES])[:, 1])

    best = max(models, key=lambda name: validation_metrics[name]["average_precision"])
    training_rows = pd.concat([train, validation], ignore_index=True)
    models[best].fit(training_rows[FEATURES], training_rows["target"])
    test_metrics = {
        "constant_baseline": evaluate(test, [0] * len(test)),
        best: evaluate(test, models[best].predict_proba(test[FEATURES])[:, 1]),
    }
    report = {
        "splits": {
            name: {
                "rows": len(split),
                "first_cutoff": str(split["cutoff"].min().date()),
                "last_cutoff": str(split["cutoff"].max().date()),
                "positive_rate": float(split["target"].mean()),
            }
            for name, split in (("train", train), ("validation", validation), ("test", test))
        },
        "selected_model": best,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    if best == "catboost":
        model_path = args.output_dir / "model.cbm"
        models[best].save_model(str(model_path))
    else:
        model_path = args.output_dir / "model.joblib"
        joblib.dump(models[best], model_path)
    report["model_file"] = model_path.name
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
