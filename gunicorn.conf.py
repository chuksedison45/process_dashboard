# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 11/18/2025
File: gunicorn.conf.py
PRODUCT: PyCharm
PROJECT: NioWave
"""

# gunicorn.conf.py
import multiprocessing

# Server socket
bind = "0.0.0.0:8050"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 300
keepalive = 2

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

# Process naming
proc_name = "cmp_dashboard"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment if using HTTPS)
# keyfile = "/path/to/your/ssl/key.pem"
# certfile = "/path/to/your/ssl/cert.pem"
