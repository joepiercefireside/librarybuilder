import os

workers = 1
threads = 1
worker_class = 'sync'
timeout = 60
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"