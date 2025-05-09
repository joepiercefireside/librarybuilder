import os

workers = 1
threads = 1
worker_class = 'gevent'
timeout = 1200
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
loglevel = 'debug'
accesslog = '-'
errorlog = '-'