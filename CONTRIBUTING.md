# Contributing

```bash
pip install -e '.[test,dev]'
pre-commit install
PYTHONPATH=src pytest tests
```

- The editable install pulls in Ruff, pytest, and other dev tools.
- `pre-commit install` keeps formatting/linting consistent with CI.
- Running pytest with `PYTHONPATH=src` mirrors the CI environment (Linux + Windows, Python 3.9/3.10/3.11).
