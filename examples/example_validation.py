import uvicorn
from pydantic import create_model

from sk_serve import SimpleAPI, serve

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

api = SimpleAPI("complete_pipeline.pkl", validation_model=model)

app = serve(api)

if __name__ == "__main__":
    uvicorn.run(
        "example_validation:app",
        host="localhost",
        port=8000,
        log_level="debug",
        reload=True,
    )
