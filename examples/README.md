## Usage Guide

### Setup

For the purposes of the guide we will use the famous "Titanic" dataset. The package is ready to use only when you have at your disposal a fitted Scikit-Learn pipeline. `produce_objects.py` will be ran to create our pipeline.

Suppose the pipeline is saved as `complete_pipeline.pkl`. After that minimal work is needed to deploy our inference endpoint (`uvincorn` is going to be needed to run the local server).

```python
import uvicorn
from sk_serve import serve, SimpleAPI

api = SimpleAPI("complete_pipeline.pkl")

app = serve(api)

if __name__ == "__main__":
    uvicorn.run("example:app", host="localhost", port=8000, log_level="debug", reload=True)
```

The code example above is identical to `example.py`, which you can run to serve the model.

### Request

After running `example.py` the `http://localhost:8000/inference` endpoint will be ready to give response to [POST] requests.

```python
import requests
import json

with open('input_data.json') as f:
    # dummy row in order to call the endpoint
    data = json.load(f)

url = "http://localhost:8000/inference"
post_response = requests.post(url, json=data)
post_response.json()

>>> {'prediction': '0'}
```

And that's it, with a couple lines of code you deploy your inference endpoint.

### Add a validation model

In order to ensure that the input data follow specific requirements you can also create a validation mode with pydantic and pass it to you SimpleAPI object.

```python
import uvicorn
from pydantic import create_model
from sk_serve import serve, SimpleAPI

model = create_model(
    "Model", pclass=(int, None), name=(str, None), sex=(str, None), age=(float, None), sibsp=(int, None), parch=(int, None),
    ticket=(str, None), fare=(float, None), cabin=(str, None), embarked=(str, None), boat=(int, None), body=(float, None),
    home=(str, None)
)

api = SimpleAPI("complete_pipeline.pkl", model)

app = serve(api)

if __name__ == "__main__":
    uvicorn.run("example_validation:app", host="localhost", port=8000, log_level="debug", reload=True)
```

The code example above is identical to `example_validation.py`, which you can run to serve the model.

Now every time you send a request the payload will be validated before it is fed into the preprocessor and/or model.
