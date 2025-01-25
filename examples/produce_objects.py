import pickle

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

x, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

x.columns = [
    "pclass",
    "name",
    "sex",
    "age",
    "sibsp",
    "parch",
    "ticket",
    "fare",
    "cabin",
    "embarked",
    "boat",
    "body",
    "home",
]

print("Dataset Loaded!")

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = LogisticRegression()

pipeline = Pipeline([("feature_preporcessing", preprocessor), ("predictor", clf)])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

trans_data = pipeline.fit_transform(x_train, y_train)

# save preprocessor
with open("complete_pipeline.pkl", "wb") as pipeline_file:
    pickle.dump(preprocessor, pipeline_file)
