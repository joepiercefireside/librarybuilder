from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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
import sqlite3
import json
import psutil
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import traceback
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session timeout
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Initialize embedding model and tokenizer lazily
tokenizer = None
model = None

# Initialize SQLite database for progress tracking
def init_progress_db():
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS progress
                 (user_id TEXT, url TEXT, links_found INTEGER, links_scanned INTEGER, items_crawled INTEGER, status TEXT)''')
    conn.commit()
    conn.close()

init_progress_db()

def load_embedding_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading all-MiniLM-L6-v2 model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
            logger.info("all-MiniLM-L6-v2 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
            raise
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

def clean_content(html):
    """Extract clean text from HTML using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())[:1000]  # Limit to 1000 chars
    except Exception as e:
        logger.error(f"Error cleaning content: {e}\n{traceback.format_exc()}")
        return html[:1000]

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

def query_grok_api(query, context):
    """Query xAI Grok API for a summary response using OpenAI client."""
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.error("XAI_API_KEY environment variable not set")
            return "Error: xAI API key not configured"
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        completion = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with expertise in analyzing web content."},
                {"role": "user", "content": f"Based on the following context, answer the question: {query}\n\nContext: {context}"}
            ],
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying xAI Grok API: {str(e)}\n{traceback.format_exc()}")
        return f"Fallback: Unable to generate AI summary due to API error. Please check API key or endpoint."

def crawl_website(start_url, user_id):
    """Crawl website using Playwright synchronously and store progress."""
    logger.info(f"Starting crawl for {start_url} by user {user_id}")
    process = psutil.Process()
    logger.info(f"Memory usage before crawl: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    visited_urls = set()
    to_visit = [start_url]
    base_domain = urllib.parse.urlparse(start_url).netloc
    links_found = 1  # Start with the initial URL
    links_scanned = 0
    items_crawled = 0
    max_items = 20  # Increased for dynamic sites
    crawled_data = []
    
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO progress (user_id, url, links_found, links_scanned, items_crawled, status) VALUES (?, ?, ?, ?, ?, ?)",
              (user_id, start_url, links_found, links_scanned, items_crawled, "running"))
    conn.commit()
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'])
            try:
                while to_visit and items_crawled < max_items:
                    url = to_visit.pop(0)
                    if url in visited_urls or not url.startswith(('http://', 'https://')):
                        logger.debug(f"Skipping invalid or visited URL: {url}")
                        continue
                    visited_urls.add(url)
                    links_scanned += 1
                    logger.debug(f"Scanning URL: {url}")
                    
                    try:
                        page = browser.new_page()
                        page.goto(url, timeout=30000)
                        page.wait_for_load_state('networkidle', timeout=30000)
                        page.wait_for_timeout(3000)  # Increased for dynamic content
                        
                        # Extract content
                        content = page.content()
                        if content:
                            embedding = generate_embedding(content)
                            if embedding is not None:
                                items_crawled += 1
                                cleaned_content = clean_content(content)
                                crawled_data.append((url, cleaned_content, embedding.tolist(), None))
                                logger.debug(f"Content found for {url}, items_crawled={items_crawled}")
                        
                        # Extract all links, including dynamic ones
                        links = page.evaluate('''() => {
                            const urls = [];
                            document.querySelectorAll('a[href], button, [role="link"], [onclick], [data-href], [data-nav], [data-url], [data-link]').forEach(el => {
                                let url = el.href || el.getAttribute('data-href') || el.getAttribute('data-nav') || el.getAttribute('data-url') || el.getAttribute('data-link');
                                if (!url && el.getAttribute('onclick')) {
                                    const match = el.getAttribute('onclick').match(/(?:location\.href|window\.open|navigateTo)\(['"]([^'"]+)['"]/);
                                    if (match) url = match[1];
                                }
                                if (url) urls.push(url);
                            });
                            return urls;
                        }''')
                        for link in links:
                            absolute_url = urllib.parse.urljoin(url, link)
                            if urllib.parse.urlparse(absolute_url).netloc == base_domain and absolute_url not in visited_urls and absolute_url not in to_visit:
                                to_visit.append(absolute_url)
                                links_found += 1
                                logger.debug(f"New link found: {absolute_url}, links_found={links_found}")
                        
                        # Update progress
                        c.execute("UPDATE progress SET links_found = ?, links_scanned = ?, items_crawled = ?, status = ? WHERE user_id = ? AND url = ?",
                                  (links_found, links_scanned, items_crawled, "running", user_id, start_url))
                        conn.commit()
                        
                        page.close()  # Close page immediately to free memory
                        logger.info(f"Memory usage after scanning {url}: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                    except Exception as e:
                        logger.error(f"Error crawling {url}: {str(e)}\n{traceback.format_exc()}")
                        c.execute("UPDATE progress SET status = ? WHERE user_id = ? AND url = ?",
                                  (f"error: {str(e)}", user_id, start_url))
                        conn.commit()
                        conn.close()
                        return crawled_data
                logger.info(f"Crawl complete: items_crawled={items_crawled}")
                c.execute("UPDATE progress SET status = ? WHERE user_id = ? AND url = ?",
                          ("complete", user_id, start_url))
                conn.commit()
            finally:
                browser.close()
                conn.close()
        return crawled_data
    except Exception as e:
        logger.error(f"Unexpected error in crawl_website: {str(e)}\n{traceback.format_exc()}")
        c.execute("UPDATE progress SET status = ? WHERE user_id = ? AND url = ?",
                  (f"error: {str(e)}", user_id, start_url))
        conn.commit()
        conn.close()
        return crawled_data

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
                login_user(User(user[0], user[1]), remember=True)
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
def crawl():
    try:
        if request.method == 'POST':
            start_url = request.form.get('url')
            if not start_url:
                flash('URL cannot be empty.', 'error')
                return render_template('crawl.html')
            
            logger.info(f"Starting crawl for {start_url} by user {current_user.id}")
            crawled_data = crawl_website(start_url, current_user.id)
            
            if crawled_data:
                conn = get_db_connection()
                cur = conn.cursor()
                query = """
                INSERT INTO documents (url, content, embedding, file_path)
                VALUES %s
                ON CONFLICT (url) DO NOTHING
                """
                execute_values(cur, query, crawled_data)
                conn.commit()
                cur.close()
                conn.close()
                flash(f"Stored {len(crawled_data)} items.", 'success')
            else:
                flash("No items crawled.", 'error')
            
            return redirect(url_for('crawl'))
        
        return render_template('crawl.html')
    except Exception as e:
        logger.error(f"Error in crawl endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error during crawl: {str(e)}", 'error')
        return render_template('crawl.html')

@app.route('/crawl_progress', methods=['GET'])
@login_required
def crawl_progress():
    try:
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        c.execute("SELECT links_found, links_scanned, items_crawled, status FROM progress WHERE user_id = ? ORDER BY rowid DESC LIMIT 1",
                  (current_user.id,))
        result = c.fetchone()
        conn.close()
        if result:
            return jsonify({
                "links_found": result[0],
                "links_scanned": result[1],
                "items_crawled": result[2],
                "status": result[3]
            })
        return jsonify({"links_found": 0, "links_scanned": 0, "items_crawled": 0, "status": "none"})
    except Exception as e:
        logger.error(f"Error in crawl_progress: {e}\n{traceback.format_exc()}")
        return jsonify({"status": f"error: {str(e)}"})

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
            
            # Generate AI summary using xAI Grok API
            context = "\n\n".join([result[1] for result in results])
            answer = query_grok_api(query, context)
            if not answer.startswith("Error") and not answer.startswith("Fallback"):
                answer = f"Answer to '{query}':\n\n{answer}\n\nRelevant Documents:"
            else:
                answer = f"Answer to '{query}':\n\n{answer}\n\nRelevant Documents:"
            
            for result in results:
                url, content, file_path, _ = result
                answer += f"\n- {content[:100]}... (Source: {url or file_path})"
            
            if not results:
                answer += "\nNo relevant content found."
            
            return render_template('search.html', results=results, query=query, answer=answer)
        
        return render_template('search.html')
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/test_playwright')
def test_playwright():
    try:
        logger.info("Starting Playwright test")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto('https://example.com')
            content = page.content()
            browser.close()
            logger.info("Playwright test completed successfully")
            return f"Playwright test successful: {len(content)} bytes"
    except Exception as e:
        logger.error(f"Playwright test failed: {str(e)}\n{traceback.format_exc()}")
        return f"Playwright test failed: {str(e)}"

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)