#!/bin/bash
pulseaudio --start -v
.venv/bin/python -m beholder.recorder record --config /home/${USER}/Desktop/beholder.ini
