from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import ValidationError
from pydantic.main import BaseModel


class SimpleAPI:
    """Simple API class that defines the HTTP endpoints for deploying a fitted scikit-learn
    pipeline. It can also take a pydantic validation model as input in order to validate the
    input every time inference is requested.

    Registered endpoints:
        - ``GET /`` — welcome message with the list of available endpoints.
        - ``GET /health`` — liveness probe (200 when a pipeline is loaded, 503 otherwise).
        - ``GET /metadata`` — model type, supported methods, feature names, load time.
        - ``POST /inference`` — single-record prediction (accepts a JSON object).
        - ``POST /batch-inference`` — batch prediction (accepts a JSON list of records).
        - ``POST /predict-proba`` — class probabilities, if the pipeline supports it.
        - ``POST /decision-function`` — decision scores, if the pipeline supports it.
    """

    def __init__(
        self,
        validation_model: type[BaseModel] | None = None,
    ):
        self.routes = APIRouter()
        self.validation_model = validation_model

        self.routes.add_api_route("/", self.home, methods=["GET"])
        self.routes.add_api_route("/health", self.health, methods=["GET"])
        self.routes.add_api_route("/metadata", self.metadata, methods=["GET"])
        self.routes.add_api_route("/inference", self.inference, methods=["POST"])
        self.routes.add_api_route(
            "/batch-inference", self.batch_inference, methods=["POST"]
        )
        self.routes.add_api_route(
            "/predict-proba", self.predict_proba, methods=["POST"]
        )
        self.routes.add_api_route(
            "/decision-function", self.decision_function, methods=["POST"]
        )

    @staticmethod
    def home() -> dict[str, str]:
        """Message returned when a GET request hits the ``/`` endpoint."""
        message = (
            "This is a simple endpoint with a deployed scikit-learn pipeline. "
            "Available endpoints: [GET] /health, [GET] /metadata, "
            "[POST] /inference, [POST] /batch-inference, "
            "[POST] /predict-proba, [POST] /decision-function."
        )
        return {"message": message}

    @staticmethod
    def health(request: Request) -> dict[str, str]:
        """Liveness probe. Returns 200 when the pipeline is loaded, 503 otherwise."""
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        return {"status": "ok"}

    @staticmethod
    def metadata(request: Request) -> dict[str, Any]:
        """Return basic information about the loaded pipeline."""
        state = request.app.state
        pipeline = getattr(state, "pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        feature_names = getattr(pipeline, "feature_names_in_", None)
        if feature_names is not None:
            feature_names = list(feature_names)

        return {
            "model_type": type(pipeline).__name__,
            "supports": getattr(state, "supports", []),
            "feature_names": feature_names,
            "loaded_at": getattr(state, "loaded_at", None),
        }

    async def inference(self, request: Request) -> dict[str, Any]:
        """Single-record inference endpoint. The body must be a JSON object whose keys
        correspond to the pipeline's expected features. The deployed pipeline must have
        a ``predict`` method.

        Args:
            request (Request): FastAPI request carrying the JSON payload.

        Returns:
            dict: ``{"prediction": <scalar>}``.
        """
        data = await request.json()
        logger.info(data)

        self._validate(data, many=False)

        x_data = pd.DataFrame(data, index=[0])
        logger.info("Getting prediction ...")
        preds = request.app.state.pipeline.predict(x_data)

        return {"prediction": preds.item()}

    async def batch_inference(self, request: Request) -> dict[str, Any]:
        """Batch inference endpoint. The body must be a JSON list of records.

        Returns:
            dict: ``{"predictions": [...]}`` with one entry per input record.
        """
        data = await request.json()
        self._require_list(data, endpoint="batch-inference")
        self._validate(data, many=True)

        x_data = pd.DataFrame(data)
        logger.info("Getting batch predictions ...")
        preds = request.app.state.pipeline.predict(x_data)

        return {"predictions": preds.tolist()}

    async def predict_proba(self, request: Request) -> dict[str, Any]:
        """Class-probability endpoint. Returns 501 if the pipeline does not expose
        ``predict_proba``. Body is a JSON list of records, like ``/batch-inference``.
        """
        pipeline = request.app.state.pipeline
        if not hasattr(pipeline, "predict_proba"):
            raise HTTPException(
                status_code=501,
                detail=f"Model '{type(pipeline).__name__}' does not support predict_proba.",
            )

        data = await request.json()
        self._require_list(data, endpoint="predict-proba")
        self._validate(data, many=True)

        x_data = pd.DataFrame(data)
        probs = pipeline.predict_proba(x_data)
        return {"probabilities": probs.tolist()}

    async def decision_function(self, request: Request) -> dict[str, Any]:
        """Decision-score endpoint. Returns 501 if the pipeline does not expose
        ``decision_function``. Body is a JSON list of records, like ``/batch-inference``.
        """
        pipeline = request.app.state.pipeline
        if not hasattr(pipeline, "decision_function"):
            raise HTTPException(
                status_code=501,
                detail=f"Model '{type(pipeline).__name__}' does not support decision_function.",
            )

        data = await request.json()
        self._require_list(data, endpoint="decision-function")
        self._validate(data, many=True)

        x_data = pd.DataFrame(data)
        scores = pipeline.decision_function(x_data)
        return {"scores": scores.tolist()}

    @staticmethod
    def _require_list(data: Any, *, endpoint: str) -> None:
        if not isinstance(data, list):
            raise HTTPException(
                status_code=422,
                detail=f"/{endpoint} expects a JSON list of records.",
            )

    def _validate(self, data: Any, *, many: bool) -> None:
        if self.validation_model is None:
            return
        logger.info("Validation of request data ...")
        try:
            if many:
                for record in data:
                    self.validation_model.model_validate(obj=record)
            else:
                self.validation_model.model_validate(obj=data)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())


def check_model_methods(model: Any, method: str) -> None:
    """Validate that ``model`` exposes a callable attribute named ``method``.

    Args:
        model: A fitted object (typically a scikit-learn estimator or pipeline).
        method (str): The attribute name to check.

    Raises:
        AttributeError: If the attribute does not exist.
        TypeError: If the attribute exists but is not callable.
    """
    try:
        attr = getattr(model, method)
    except AttributeError as e:
        logger.error(e)
        raise

    if not callable(attr):
        raise TypeError(
            f"'{method}' on {type(model).__name__} exists but is not callable."
        )
