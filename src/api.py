from fastapi import APIRouter
import inspect
from pydantic.main import BaseModel
from sklearn.compose import ColumnTransformer
import pickle
import pandas as pd

class InfData(BaseModel):
    pclass: int
    name: str
    sex: str
    age: float
    sibsp: int
    parch: int
    ticket: str
    fare: float
    cabin: str
    embarked: str
    boat: int
    body: float
    home: str

class SimpleAPI():
    def __init__(self, pipeline_path : str, model_path: str):
        self.app = APIRouter()
        self.pipeline_path = pipeline_path
        self.model_path = model_path
        for method, _ in inspect.getmembers(self, inspect.ismethod):
            if not method.startswith("_"):
                self.app.add_api_route(f"/{method}", getattr(self, method), methods=["POST"])
                
    def home(self):
        home_message = f"This is a simple endpoint with a deployed scikit-learn model and pipeline. \
        Only available endpoints are: /home and /inference."
        
        return {"home_message": home_message}

    def inference(self, inf_data: InfData):
        x_data = pd.DataFrame(inf_data.model_dump(), index=[0])
        with open(self.pipeline_path, 'rb') as pipeline_file:
            pipeline = pickle.load(pipeline_file)
            # make sure the pickle we loaded is a Pipeline object
            assert isinstance(pipeline, ColumnTransformer), "ColumnTransformer object loaded is not a `sklearn.compose.ColumnTransformer` object."
        with open(self.model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            # make sure the pickle loaded has predict and predict_proba methods
            try:
                self._check_model_methods(model, "predict")
            except Exception as e:
                print(e)
                raise RuntimeError("The object that was loaded doesn't have `predict` method.")
                
        trans_data = pipeline.transform(x_data)
        preds = model.predict(trans_data)

        return {"prediction": int(preds.item())}

    @staticmethod
    def _check_model_methods(model, method: str):
        try:
           method_name = getattr(model, method)
        except Exception as e:
            raise(e)

        assert callable(method_name)
