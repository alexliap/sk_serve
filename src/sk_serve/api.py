from typing import Dict, Union

import pandas as pd
from fastapi import APIRouter, Request
from loguru import logger
from pydantic.main import BaseModel


class SimpleAPI:
    """Simple API class that takes pipeline/model path as arguments and defines one inference endpoint for
    simple model deployment. The pipeline must be a Scikit-Learn Pipeline. It can also take a pydantic validation model
    as input in order to validate the input everytime inference is requested.
    """

    def __init__(
        self,
        validation_model: Union[type[BaseModel], None] = None,
    ):
        self.routes = APIRouter()
        self.validation_model = validation_model

        # add our only 2 endpoints
        self.routes.add_api_route("/", getattr(self, "home"), methods=["GET"])
        self.routes.add_api_route(
            "/inference", getattr(self, "inference"), methods=["POST"]
        )

    @staticmethod
    def home() -> Dict[str, str]:
        """Method that returns a message when sending a GET request to the `/` endpoint."""
        home_message = (
            "This is a simple endpoint with a deployed scikit-learn pipeline. \
            Only available endpoints is: [POST] /inference."
        )

        return {"message": home_message}

    async def inference(self, request: Request):
        """Inference method that is used by the inference endpoint. In order to get the prediction
        the deployed pipeline must have the `predict` method.

        Args:
            inf_data (dict): Input data for inference. Currently only one data point at a time is supported.

        Returns:
            dict: The prediction.
        """
        data = await request.json()

        logger.info(data)

        if self.validation_model is not None:
            logger.info("Validation of requerst data ...")
            self.validation_model.model_validate(obj=data)

        x_data = pd.DataFrame(data, index=[0])

        # get predictions
        logger.info("Getting prediction ...")
        preds = request.app.state.pipeline.predict(x_data)

        return {"prediction": preds.item()}


def check_model_methods(model, method: str):
    """Helper function that checks if a class method exits or not.

    Args:
        model: A Scikit-learn model.
        method (str): The name of the respective method.
    """
    try:
        method_name = getattr(model, method)
    except Exception as e:
        logger.error(e)
        raise e

    assert callable(method_name)
