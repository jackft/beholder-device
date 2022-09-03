import pytest

import pathlib
import subprocess

from typing import List

@pytest.fixture
def vpath() -> pathlib.Path:
    path = pathlib.Path("example.mp4")
    url = "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
    if not path.exists():
        subprocess.run(["curl", url, "--output", path])
    return path