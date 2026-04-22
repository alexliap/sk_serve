# Changelog

## 1.2.0

Backward-compatible. Existing `SimpleAPI()` + `serve(api)` code continues to work.

### Added

- `POST /batch-inference` — accepts a JSON list of records and returns a list of predictions.
- `POST /predict-proba` — returns class probabilities for classifiers; responds with `501 Not Implemented` when the pipeline does not expose `predict_proba`.
- `POST /decision-function` — returns decision scores; responds with `501 Not Implemented` when the pipeline does not expose `decision_function`.
- `GET /health` — liveness probe (`200` when the pipeline is loaded, `503` otherwise).
- `GET /metadata` — reports the model type, supported prediction methods, feature names (when available via `feature_names_in_`), and load time.
- `joblib` support — models saved with a `.joblib` extension are loaded via `joblib.load`; `.pkl` / `.pickle` continue to use `pickle.load`.
- `serve(simple_api, pipeline=...)` — optional keyword argument to inject a fitted pipeline directly. When provided, `MODEL_PATH` is ignored. Useful for in-process use and tests.
- Minimal pytest suite under `tests/` covering the endpoints and backward-compat path.

### Fixed

- Application no longer starts in a broken state when the loaded object is missing `predict`; the startup error now propagates instead of being logged and swallowed.
- `check_model_methods` no longer relies on `assert`, which was optimized away under `python -O`. It now raises `TypeError` when the attribute exists but is not callable.
- Validation errors raised by the supplied pydantic model are now returned as `HTTP 422` instead of propagating as `HTTP 500`.
- Typo in the inference log line (`"requerst"` → `"request"`).

### Changed

- The `/` welcome message now lists all available endpoints.
- Type hints modernized for Python 3.12 (`X | None`, `dict[str, str]`).
- `joblib` added as an explicit runtime dependency (already a transitive dep of `scikit-learn`, so no new install footprint).
- `pytest` and `httpx` added to the `dev` optional dependencies for running the new test suite.
