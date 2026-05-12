import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from features import FEATURES


class PurchaseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    order_count: int = Field(ge=1, strict=True)
    total_spend: float = Field(gt=0)
    mean_order_value: float = Field(gt=0)
    days_since_last_order: float = Field(gt=0, le=90)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("RETAIL_MODEL_PATH", "models/repeat-purchase.cbm"))
    if model_path.suffix == ".cbm":
        model = CatBoostClassifier()
        model.load_model(str(model_path))
    elif model_path.suffix == ".joblib":
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    app.state.model = model
    yield


app = FastAPI(title="Repeat Purchase Prediction", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: PurchaseFeatures, request: Request) -> dict[str, float]:
    row = pd.DataFrame([features.model_dump()], columns=FEATURES)
    score = request.app.state.model.predict_proba(row)[0, 1]
    return {"repeat_purchase_probability": float(score)}
