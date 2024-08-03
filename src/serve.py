from fastapi import FastAPI
import uvicorn
from api import SimpleAPI

def serve(pipeline_path : str, model_path: str):
    app = FastAPI()
    dummy = SimpleAPI(pipeline_path, model_path)
    app.include_router(dummy.app)

    return app

app = serve("pipeline.pkl", "model.pkl")

def start_server():
    uvicorn.run("main:app", host="localhost", port=8000, log_level="debug", reload=True)

if __name__ == '__main__':
    start_server()