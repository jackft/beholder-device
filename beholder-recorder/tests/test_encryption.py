import pytest

import subprocess
import pathlib
import tempfile

def test_encryption(vpath: pathlib.Path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        pass