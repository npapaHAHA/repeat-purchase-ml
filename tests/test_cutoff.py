import sqlite3
import unittest
from contextlib import closing

import pandas as pd

from train import SNAPSHOT_QUERY


class CutoffTest(unittest.TestCase):
    def test_future_orders_only_affect_target(self):
        rows = [
            (1, "2011-02-01 00:00:00", 15.0),
            (1, "2011-02-28 12:00:00", 25.0),
            (1, "2011-03-01 00:00:00", 500.0),
            (2, "2011-02-01 00:00:00", 8.0),
            (2, "2011-03-31 00:00:00", 30.0),
            (3, "2011-03-02 00:00:00", 40.0),
        ]
        orders = pd.DataFrame(rows, columns=["customer_id", "ordered_at", "order_value"])
        parameters = {
            "history_start": "2010-12-01 00:00:00",
            "cutoff": "2011-03-01 00:00:00",
            "outcome_end": "2011-03-31 00:00:00",
        }
        with closing(sqlite3.connect(":memory:")) as connection:
            orders.to_sql("orders", connection, index=False)
            result = pd.read_sql_query(SNAPSHOT_QUERY, connection, params=parameters)
        self.assertEqual(result["customer_id"].tolist(), [1, 2])
        self.assertEqual(result["order_count"].tolist(), [2, 1])
        self.assertEqual(result["total_spend"].tolist(), [40.0, 8.0])
        self.assertEqual(result["target"].tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
