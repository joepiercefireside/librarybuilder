from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from psycopg2.extras import execute_values
from werkzeug.security import generate_password_hash, check_password_hash
import os
import logging
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
import urllib.parse
import asyncio
import json
import psutil
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy import Spider
import traceback

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Initialize embedding model and tokenizer lazily
tokenizer = None
model = None

def load_embedding_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading all-MiniLM-L6-v2 model...")
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("all-MiniLM-L6-v2 model loaded.")
    return tokenizer, model

def unload_embedding_model():
    global tokenizer, model
    tokenizer = None
    model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info("Unloaded all-MiniLM-L6-v2 model to free memory.")

# Database connection function with error handling
def get_db_connection():
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        logger.info("Database connection established.")
        return conn
    except KeyError as e:
        logger.error(f"Missing DATABASE_URL environment variable: {e}")
        raise
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user:
            logger.info(f"User loaded: ID {user_id}")
            return User(user[0], user[1])
        logger.warning(f"No user found for ID {user_id}")
        return None
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {e}")
        return None

def generate_embedding(text):
    """Generate embedding for a single text."""
    try:
        process = psutil.Process()
        logger.info(f"Memory usage before embedding: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        tokenizer, model = load_embedding_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        unload_embedding_model()  # Free memory after embedding
        logger.info(f"Memory usage after embedding: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}\n{traceback.format_exc()}")
        return None

class WebsiteSpider(Spider):
    name = 'website_spider'
    
    def __init__(self, start_url, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.base_domain = urllib.parse.urlparse(start_url).netloc
        self.visited_urls = set()
        self.links_found = 1  # Start with the initial URL
        self.links_scanned = 0
        self.items_crawled = 0
        self.max_items = 10  # Limit for testing
        self.progress = []

    def parse(self, response):
        self.links_scanned += 1
        self.visited_urls.add(response.url)
        logger.debug(f"Scanning URL: {response.url}")
        
        try:
            # Extract content (using Playwright for dynamic content)
            content = response.css('body').get(default='').strip()
            if content:
                self.items_crawled += 1
                embedding = generate_embedding(content)
                if embedding is not None:
                    yield {
                        'url': response.url,
                        'content': content,
                        'embedding': embedding.tolist(),
                        'file_path': None
                    }
                else:
                    logger.warning(f"No embedding generated for {response.url}")
                
                self.progress.append({
                    "links_found": self.links_found,
                    "links_scanned": self.links_scanned,
                    "items_crawled": self.items_crawled
                })
            
            # Follow links (same domain)
            for link in response.css('a::attr(href)').getall():
                absolute_url = response.urljoin(link)
                if urllib.parse.urlparse(absolute_url).netloc == self.base_domain and absolute_url not in self.visited_urls and self.items_crawled < self.max_items:
                    self.links_found += 1
                    self.visited_urls.add(absolute_url)
                    logger.debug(f"New link found: {absolute_url}, links_found={self.links_found}")
                    yield response.follow(
                        absolute_url,
                        callback=self.parse,
                        meta={'playwright': True}  # Enable Playwright for dynamic content
                    )
            
            self.progress.append({
                "links_found": self.links_found,
                "links_scanned": self.links_scanned,
                "items_crawled": self.items_crawled
            })
        except Exception as e:
            logger.error(f"Error parsing {response.url}: {str(e)}\n{traceback.format_exc()}")
            self.progress.append({
                "status": "error",
                "message": str(e)
            })

@app.route('/')
def index():
    try:
        logger.info("Accessing index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                flash('Email already registered.', 'error')
            else:
                password_hash = generate_password_hash(password)
                cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password_hash))
                conn.commit()
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            cur.close()
            conn.close()
        return render_template('register.html')
    except Exception as e:
        logger.error(f"Error in register endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT id, email, password_hash FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()
            conn.close()
            if user and check_password_hash(user[2], password):
                login_user(User(user[0], user[1]))
                logger.info(f"User {email} logged in successfully")
                return redirect(url_for('index'))
            flash('Invalid email or password.', 'error')
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error in login endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        logger.info("User logged out")
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error in logout endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/crawl', methods=['GET', 'POST'])
@login_required
async def crawl():
    try:
        if request.method == 'POST':
            start_url = request.form.get('url')
            if not start_url:
                flash('URL cannot be empty.', 'error')
                return render_template('crawl.html')
            
            logger.info(f"Crawling website: {start_url}")
            # Initialize Scrapy settings
            settings = get_project_settings()
            settings.set('ITEM_PIPELINES', {})
            settings.set('PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT', 30000)  # 30 seconds
            settings.set('DOWNLOAD_HANDLERS', {
                'http': 'scrapy_playwright.handler.PlaywrightHandler',
                'https': 'scrapy_playwright.handler.PlaywrightHandler',
            })
            settings.set('TWISTED_REACTOR', 'twisted.internet.asyncioreactor.AsyncioSelectorReactor')
            
            # Run Scrapy spider
            process = CrawlerProcess(settings)
            spider = WebsiteSpider(start_url=start_url)
            process.crawl(spider)
            
            async def generate():
                try:
                    # Start Scrapy process
                    process.start()
                    
                    # Stream progress updates via SSE
                    for update in spider.progress:
                        yield f"data: {json.dumps(update)}\n\n"
                    
                    # Store crawled data in database
                    if spider.items_crawled > 0:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        query = """
                        INSERT INTO documents (url, content, embedding, file_path)
                        VALUES %s
                        ON CONFLICT (url) DO NOTHING
                        """
                        execute_values(cur, query, [
                            (item['url'], item['content'], item['embedding'], item['file_path'])
                            for item in spider.parsed_items
                        ])
                        conn.commit()
                        cur.close()
                        conn.close()
                        yield f"data: {json.dumps({'status': 'stored', 'items_crawled': spider.items_crawled})}\n\n"
                    else:
                        yield f"data: {json.dumps({'status': 'complete', 'items_crawled': 0})}\n\n"
                except Exception as e:
                    logger.error(f"Error in Scrapy crawl: {str(e)}\n{traceback.format_exc()}")
                    yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        
        return render_template('crawl.html')
    except Exception as e:
        logger.error(f"Error in crawl endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    try:
        if request.method == 'POST':
            query = request.form.get('query')
            if not query:
                flash('Query cannot be empty.', 'error')
                return render_template('search.html')
            
            logger.info(f"Search query: {query}")
            query_embedding = generate_embedding(query)
            if query_embedding is None:
                flash('Failed to generate query embedding.', 'error')
                return render_template('search.html')
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
            SELECT url, content, file_path, embedding
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT 5
            """, (query_embedding.tolist(),))
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            # Mock LLM answer
            answer = f"Answer to '{query}':\n\n"
            for result in results:
                url, content, file_path, _ = result
                answer += f"- {content[:100]}... (Source: {url or file_path})\n"
            
            return render_template('search.html', results=results, query=query, answer=answer)
        
        return render_template('search.html')
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/test_playwright')
async def test_playwright():
    try:
        logger.info("Starting Playwright test")
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto('https://example.com')
            content = await page.content()
            await browser.close()
            logger.info("Playwright test completed successfully")
            return f"Playwright test successful: {len(content)} bytes"
    except Exception as e:
        logger.error(f"Playwright test failed: {str(e)}\n{traceback.format_exc()}")
        return f"Playwright test failed: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)