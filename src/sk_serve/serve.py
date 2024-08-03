# import argparse
# import uvicorn
from api import SimpleAPI
from fastapi import FastAPI


def serve(pipeline_path: str, model_path: str):
    """_summary_

    Args:
        pipeline_path (str): _description_
        model_path (str): _description_

    Returns:
        _type_: _description_
    """
    app = FastAPI()
    api = SimpleAPI(pipeline_path, model_path)
    app.include_router(api.routes)

    return app


# def start_server(host: str = "localhost"):
#     uvicorn.run("serve:app", host=host, port=8000, log_level="debug", reload=True)

# if __name__ == '__main__':
#     start_server()
