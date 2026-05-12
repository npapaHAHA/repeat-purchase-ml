import sqlite3
from contextlib import closing
from io import BytesIO
from zipfile import ZipFile

import pandas as pd


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
