from fastapi import FastAPI

from .api import SimpleAPI


def serve(simple_api: SimpleAPI):
    """Function that constructs the model API.

    Args:
        simple_api (SimpleAPI): The SimpleAPI object needed for deployment.

    Returns:
        app (FastAPI): The FastAPI application.
    """
    app = FastAPI()
    api = simple_api
    app.include_router(api.routes)

    return app
