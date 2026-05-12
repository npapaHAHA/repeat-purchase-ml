import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "example-request.json").read_text(encoding="utf-8"))


class ApiTest(unittest.TestCase):
    def setUp(self):
        model = ROOT / "models/repeat-purchase.cbm"
        self.enterContext(patch.dict(os.environ, {"RETAIL_MODEL_PATH": str(model)}))
        self.client = self.enterContext(TestClient(app))

    def test_health_after_loading_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_returns_probability_from_bundled_model(self):
        response = self.client.post("/predict", json=EXAMPLE)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertAlmostEqual(response.json()["repeat_purchase_probability"], 0.420565, places=5)

    def test_rejects_invalid_features(self):
        invalid = (
            {"order_count": 0},
            {"order_count": True},
            {"order_count": "3"},
            {"days_since_last_order": 91},
            {"extra_field": 1},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                response = self.client.post("/predict", json={**EXAMPLE, **changes})
                self.assertEqual(response.status_code, 422, response.text)

    def test_rejects_missing_feature(self):
        payload = {name: value for name, value in EXAMPLE.items() if name != "total_spend"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422, response.text)
