import uvicorn
from dotenv import load_dotenv
from pydantic import create_model

from sk_serve import SimpleAPI, serve

load_dotenv()

model = create_model(
    "Model",
    pclass=(int, None),
    name=(str, None),
    sex=(str, None),
    age=(float, None),
    sibsp=(int, None),
    parch=(int, None),
    ticket=(str, None),
    fare=(float, None),
    cabin=(str, None),
    embarked=(str, None),
    boat=(int, None),
    body=(float, None),
    home=(str, None),
)

api = SimpleAPI(validation_model=model)

app = serve(api)

if __name__ == "__main__":
    uvicorn.run(
        "example_validation:app",
        host="localhost",
        port=8000,
        log_level="debug",
        reload=True,
    )
