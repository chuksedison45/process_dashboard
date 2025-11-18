#!/bin/bash
# start_server.sh

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start Gunicorn server
echo "Starting CMP Dashboard Server..."
gunicorn --config gunicorn.conf.py wsgi:application