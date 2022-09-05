#!/bin/bash
# start.sh

export FLASK_APP=wsgi.py
export FLASK_DEBUG=1
export APP_CONFIG_FILE=config.py


.venv/bin/flask init-db
.venv/bin/uwsgi --ini uwsgi.ini 2>&1
