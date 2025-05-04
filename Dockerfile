FROM python:3.11.9-slim

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcairo2 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libpango-1.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers with detailed logging
RUN echo "Installing Playwright..." && \
    python -m playwright install && \
    python -m playwright install-deps && \
    python -m playwright --version && \
    ls -l /root/.cache/ms-playwright || { echo "Playwright binaries not found"; exit 1; } && \
    find /root/.cache/ms-playwright -type f -exec ls -l {} \; && \
    echo "Playwright installation completed."

# Copy application code
COPY . .

# Run the app
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]