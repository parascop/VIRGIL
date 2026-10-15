import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from helpers import make_profile


@pytest.fixture
def profile(tmp_path):
    return make_profile(tmp_path)
