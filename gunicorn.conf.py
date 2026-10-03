"""Toolforge's proxy connects to port 8000 inside the container."""

import os

from cpus import compute_cpus

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = compute_cpus()
threads = 1
worker_class = "gthread"
timeout = 300
graceful_timeout = 30

keepalive = 0
accesslog = "-"
errorlog = "-"
# loglevel = "debug"
capture_output = True
