import pickle
from typing import Dict, Union

import pandas as pd
from fastapi import APIRouter
from pydantic.main import BaseModel
from sklearn.compose import ColumnTransformer


class SimpleAPI:
    """Simple API class that takes pipeline and model paths as arguments and defines one inference endpoint for
    simple model deployment. Both loaded object must be Scikit-learn objects. It can also take a pydantic validation model
    as input in order to validate the input everytime inference is requested.
    """

    def __init__(
        self,
        pipeline_path: str,
        model_path: str,
        validation_model: Union[BaseModel, None] = None,
    ):
        self.routes = APIRouter()
        self.pipeline_path = pipeline_path
        self.model_path = model_path
        self.validation_model = None

        if validation_model is not None:
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
            "This is a simple endpoint with a deployed scikit-learn model and pipeline. \
Only available endpoints is: [POST] /inference."
        )

        return {"message": home_message}

    def inference(self, inf_data: dict):
        """Inference method that is used by the inference endpoint. In order to get the prediction
        two checks are made beforehand: check if the pipeline is a `sklearn.compose.ColumnTransformer` object &
        if the model loaded has `predict` method.

        Args:
            inf_data (dict): Input data for inference. Currently only one data point at a time is supported.

        Raises:
            RuntimeError: If the model loaded doesn't have `predict` method.

        Returns:
            dict: The prediction.
        """
        if self.validation_model is not None:
            self.validation_model.model_validate(inf_data)

        x_data = pd.DataFrame(inf_data, index=[0])

        with open(self.pipeline_path, "rb") as pipeline_file:
            pipeline = pickle.load(pipeline_file)
            # make sure the pickle we loaded is a Pipeline object
            assert isinstance(
                pipeline, ColumnTransformer
            ), "ColumnTransformer object loaded is not a `sklearn.compose.ColumnTransformer` object."

        with open(self.model_path, "rb") as model_file:
            model = pickle.load(model_file)
            # make sure the pickle loaded has predict and predict_proba methods
            try:
                self._check_model_methods(model, "predict")
            except Exception as e:
                print(e)
                raise RuntimeError(
                    "The object that was loaded doesn't have `predict` method."
                )

        # apply column transforms
        trans_data = pipeline.transform(x_data)
        # get predictions
        preds = model.predict(trans_data)

        return {"prediction": preds.item()}

    @staticmethod
    def _check_model_methods(model, method: str):
        """Helper function that checks if a class method exits or not.

        Args:
            model: A Scikit-learn model.
            method (str): The name of the respective method.
        """
        try:
            method_name = getattr(model, method)
        except Exception as e:
            raise (e)

        assert callable(method_name)
