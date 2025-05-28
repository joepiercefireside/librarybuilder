FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt . RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl \
libnss3 libxss1 libasound2 libatk1.0-0 libatk-bridge2.0-0 libcairo2 \
libcups2 libdrm2 libgbm1 libgtk-3-0 libpango-1.0-0 libxcomposite1 \
libxdamage1 libxfixes3 libxrandr2 libxtst6 && \
curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
apt-get install -y nodejs && \
apt-get clean && rm -rf /var/lib/apt/lists/\*

COPY package.json . RUN npm install

COPY . .

ENV PORT=8000 EXPOSE $PORT

CMD \["gunicorn", "--config", "gunicorn.conf.py", "app:app"\]