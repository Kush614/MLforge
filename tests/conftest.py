import os
import tempfile
import pytest

# Tests write their metrics to a throwaway dir so they never clobber runs/latest.jsonl.
os.environ["AGENTFORGE_RUNS_DIR"] = tempfile.mkdtemp(prefix="agentforge-test-runs-")

# Tests never upload traces or call LLMs, whatever is in the developer's .env.
# Empty strings (not pops): load_dotenv() never overrides an existing key, so a developer's
# .env cannot leak real credentials into the test run.
os.environ["WEAVE_DISABLED"] = "true"
for _k in ("WANDB_API_KEY", "WANDB_ENTITY", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_MODEL"):
    os.environ[_k] = ""


@pytest.fixture
def synthetic_xy():
    import numpy as np
    from agentforge import data as d
    df = d.synthetic_site("cleveland")
    df["target"] = (df["target"] > 0).astype(int)
    X = df[d.FEATURES].values.astype(float)
    y = df["target"].values
    return X[:220], y[:220], X[220:], y[220:]
