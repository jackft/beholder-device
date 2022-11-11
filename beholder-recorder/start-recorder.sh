#!/bin/bash
# start-recorder.sh
pulseaudio --start -v
.venv/bin/python -m beholder.recorder record --config /home/${USER}/Desktop/beholder.ini --log-level debug
