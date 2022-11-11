#!/bin/bash
# start-webapp.sh

export FLASK_APP=wsgi.py
export FLASK_DEBUG=1
export APP_CONFIG_FILE=config.py
mkdir -p /tmp/test
.venv/bin/uwsgi --ini uwsgi.ini 2>&1
