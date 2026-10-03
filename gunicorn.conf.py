import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = 1
timeout = 180
graceful_timeout = 30
