import os
import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI
from loguru import logger

from .api import SimpleAPI, check_model_methods

_PREDICTION_METHODS = ("predict", "predict_proba", "decision_function")


def _load_model(path: str) -> Any:
    ext = Path(path).suffix.lower()
    if ext == ".joblib":
        logger.info(f"Loading model via joblib from {path}")
        return joblib.load(path)
    logger.info(f"Loading model via pickle from {path}")
    with open(path, "rb") as model_file:
        return pickle.load(model_file)


def _register_pipeline(app: FastAPI, pipeline: Any) -> None:
    check_model_methods(pipeline, "predict")
    app.state.pipeline = pipeline
    app.state.supports = [
        m
        for m in _PREDICTION_METHODS
        if hasattr(pipeline, m) and callable(getattr(pipeline, m))
    ]
    app.state.loaded_at = datetime.now(timezone.utc).isoformat()
    logger.info("✅ Model loaded")


def _make_lifespan(pipeline: Any | None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            _register_pipeline(app, pipeline)
        else:
            model_path = os.getenv("MODEL_PATH")
            if model_path is None:
                raise RuntimeError("MODEL_PATH environment variable does not exist.")
            loaded = _load_model(model_path)
            _register_pipeline(app, loaded)
        yield
        logger.info("👋 Shutting down ...")

    return lifespan


def serve(simple_api: SimpleAPI, pipeline: Any | None = None) -> FastAPI:
    """Function that constructs the model API.

    Args:
        simple_api (SimpleAPI): The SimpleAPI object needed for deployment.
        pipeline: Optional fitted pipeline to inject directly. When provided, the
            ``MODEL_PATH`` environment variable is ignored and no file is read.
            Useful for unit tests and for callers that already have the pipeline
            in memory.

    Returns:
        app (FastAPI): The FastAPI application.
    """
    app = FastAPI(lifespan=_make_lifespan(pipeline))
    app.include_router(simple_api.routes)
    return app
