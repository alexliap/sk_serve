import uvicorn

from sk_serve import SimpleAPI, serve

api = SimpleAPI("complete_pipeline.pkl")

app = serve(api)

if __name__ == "__main__":
    uvicorn.run(
        "example:app", host="localhost", port=8000, log_level="debug", reload=True
    )
