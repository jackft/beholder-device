import pytest

import datetime
import pathlib

import beholder.recorder.recorder

def test_simple_recorder():
    recorder = beholder.recorder.recorder.Recorder.from_file(pathlib.Path("beholder.ini"))
    recorder.record_script(now=True)
    assert True
