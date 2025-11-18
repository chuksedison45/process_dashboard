# -*- coding: utf-8 -*-
"""
@author: Edison Chukwuemeka
@date: 11/18/2025
File: wsgi.py
PRODUCT: PyCharm
PROJECT: NioWave
"""


# wsgi.py
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from cmp_process_dashboard import app

# Expose the Flask app for WSGI
application = app.server

if __name__ == "__main__":
    application.run()
