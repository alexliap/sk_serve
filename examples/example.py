import uvicorn
from dotenv import load_dotenv
from sk_serve import SimpleAPI, serve

load_dotenv()

api = SimpleAPI()

app = serve(api)

if __name__ == "__main__":
    uvicorn.run(
        "example:app", host="localhost", port=8000, log_level="debug", reload=True
    )
