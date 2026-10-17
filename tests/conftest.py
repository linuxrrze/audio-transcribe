import os

import pytest

# Tests compare English UI text; a German system locale must not change it.
os.environ["TRANSCRIBE_LANG"] = "en"


def pytest_collection_modifyitems(config, items):
    if os.environ.get("TX_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="needs model downloads; set TX_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
