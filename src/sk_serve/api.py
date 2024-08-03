import pickle
from typing import Dict, Union

import pandas as pd
from fastapi import APIRouter
from pydantic.main import BaseModel
from sklearn.compose import ColumnTransformer


class SimpleAPI:
    """_summary_"""

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
        """Method that returns a message when accessing the `/` endpoint."""
        home_message = (
            "This is a simple endpoint with a deployed scikit-learn model and pipeline. \
Only available endpoints is: [POST] /inference."
        )

        return {"message": home_message}

    def inference(self, inf_data: dict):
        """_summary_

        Args:
            inf_data (dict): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
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

        return {"prediction": int(preds.item())}

    @staticmethod
    def _check_model_methods(model, method: str):
        """_summary_

        Args:
            model (_type_): _description_
            method (str): _description_
        """
        try:
            method_name = getattr(model, method)
        except Exception as e:
            raise (e)

        assert callable(method_name)
