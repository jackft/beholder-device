#!/bin/bash
pulseaudio --start -v
.venv/bin/python -m beholder.recorder record --config /home/beholder/Desktop/beholder.ini
