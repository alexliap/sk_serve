from fastapi import FastAPI
import pickle
import os
from loguru import logger
from contextlib import asynccontextmanager
from .api import SimpleAPI, check_model_methods


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH")

    if model_path is None:
        raise RuntimeError("MODEL_PATH environment variable does not exist.")

    with open(model_path, "rb") as model_file:
        logger.info("Loading deployed model ...")
        app.state.pipeline = pickle.load(model_file)
        try:
            check_model_methods(app.state.pipeline, "predict")
            logger.info("✅ Model loaded")
        except Exception as e:
            message = "The object that was loaded doesn't have `predict` method"
            logger.error(f"{message}: -> {e}")
    yield
    
    logger.info("👋 Shutting down...")


def serve(simple_api: SimpleAPI):
    """Function that constructs the model API.

    Args:
        simple_api (SimpleAPI): The SimpleAPI object needed for deployment.

    Returns:
        app (FastAPI): The FastAPI application.
    """
    app = FastAPI(lifespan=lifespan)
    api = simple_api
    app.include_router(
        api.routes,
    )
    return app
