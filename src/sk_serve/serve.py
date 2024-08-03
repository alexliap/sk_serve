from fastapi import FastAPI

from .api import SimpleAPI


def serve(simple_api: SimpleAPI):
    """_summary_

    Args:
        pipeline_path (str): _description_
        model_path (str): _description_

    Returns:
        _type_: _description_
    """
    app = FastAPI()
    api = simple_api
    app.include_router(api.routes)

    return app
