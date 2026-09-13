import os
import pytest

# Tests never upload traces or call LLMs, whatever is in the developer's .env.
os.environ["WEAVE_DISABLED"] = "true"
os.environ.pop("WANDB_API_KEY", None)
os.environ.pop("TYPESAFE_API_KEY", None)
os.environ.pop("TYPESAFE_BASE_URL", None)


@pytest.fixture
def synthetic_xy():
    import numpy as np
    from agentforge import data as d
    df = d.synthetic_site("cleveland")
    df["target"] = (df["target"] > 0).astype(int)
    X = df[d.FEATURES].values.astype(float)
    y = df["target"].values
    return X[:220], y[:220], X[220:], y[220:]
