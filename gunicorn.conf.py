import os

workers = 1
threads = 1
worker_class = 'gevent'
timeout = 1800
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
loglevel = 'debug'
accesslog = '-'
errorlog = '-'
worker_tmp_dir = "/dev/shm"  # Use shared memory for worker temp files