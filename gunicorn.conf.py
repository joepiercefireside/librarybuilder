import os

workers = 2  # Increased for better concurrency
threads = 4  # Increased for async task handling
worker_class = 'gevent'
timeout = 1800  # 30 minutes, sufficient for long-running crawls
keepalive = 5  # Keep connections alive for 5s
limit_request_line = 4094  # Max size of HTTP request line
limit_request_fields = 100  # Max number of header fields
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
loglevel = 'debug'
accesslog = '-'
errorlog = '-'
worker_tmp_dir = "/dev/shm"  # Use shared memory for worker temp files