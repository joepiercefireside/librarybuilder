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
from crawl4ai import AsyncWebCrawler
import urllib.parse
import urllib.request
import asyncio
import json

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        tokenizer, model = load_embedding_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

async def crawl_website_progress(start_url):
    """Stream progress updates for website crawling."""
    async with AsyncWebCrawler(verbose=True) as crawler:
        visited_urls = set()
        to_visit = [start_url]
        base_domain = urllib.parse.urlparse(start_url).netloc
        links_found = 1  # Start with the initial URL
        links_scanned = 0
        items_crawled = 0
        
        while to_visit and items_crawled < 10:  # Reduced limit for testing
            url = to_visit.pop(0)
            if url in visited_urls:
                continue
            visited_urls.add(url)
            links_scanned += 1
            
            try:
                result = await crawler.arun(url=url, follow_links=True, max_depth=2)
                if result.success:
                    content = result.markdown
                    if content:
                        items_crawled += 1
                    
                    # Add new links to visit (same domain)
                    for link in result.links:
                        if urllib.parse.urlparse(link).netloc == base_domain and link not in visited_urls and link not in to_visit:
                            to_visit.append(link)
                            links_found += 1
                    
                    # Yield progress update
                    yield json.dumps({
                        "links_found": links_found,
                        "links_scanned": links_scanned,
                        "items_crawled": items_crawled
                    }) + "\n"
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
        
        yield json.dumps({"status": "complete", "items_crawled": items_crawled}) + "\n"

async def crawl_website_data(start_url):
    """Collect crawled data for storage."""
    async with AsyncWebCrawler(verbose=True) as crawler:
        crawled_data = []
        visited_urls = set()
        to_visit = [start_url]
        base_domain = urllib.parse.urlparse(start_url).netloc
        
        while to_visit and len(crawled_data) < 10:  # Reduced limit for testing
            url = to_visit.pop(0)
            if url in visited_urls:
                continue
            visited_urls.add(url)
            
            try:
                result = await crawler.arun(url=url, follow_links=True, max_depth=2)
                if result.success:
                    content = result.markdown
                    if content:
                        embedding = generate_embedding(content)
                        if embedding is not None:
                            crawled_data.append((url, content, embedding.tolist(), None))
                        else:
                            logger.warning(f"No embedding generated for {url}")
                    
                    # Extract PDFs (placeholder for downloading)
                    for link in result.links:
                        if link.endswith('.pdf'):
                            pdf_path = f"data/pdfs/{urllib.parse.quote(link, safe='')}.pdf"
                            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                            try:
                                with urllib.request.urlopen(link) as response, open(pdf_path, 'wb') as out_file:
                                    out_file.write(response.read())
                                pdf_text = extract_pdf_text(pdf_path)  # Placeholder
                                if pdf_text:
                                    embedding = generate_embedding(pdf_text)
                                    if embedding is not None:
                                        crawled_data.append((link, pdf_text, embedding.tolist(), pdf_path))
                                    else:
                                        logger.warning(f"No embedding generated for PDF {link}")
                            except Exception as e:
                                logger.error(f"Error downloading PDF {link}: {e}")
                    
                    # Add new links to visit (same domain)
                    for link in result.links:
                        if urllib.parse.urlparse(link).netloc == base_domain and link not in visited_urls and link not in to_visit:
                            to_visit.append(link)
                    
                    # Free memory
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
        
        return crawled_data

def extract_pdf_text(pdf_path):
    """Placeholder for PDF text extraction."""
    return ""  # Implement using pdfplumber later

@app.route('/')
def index():
    try:
        logger.info("Accessing index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
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
        logger.error(f"Error in register endpoint: {e}")
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
        logger.error(f"Error in login endpoint: {e}")
        return "Internal Server Error", 500

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        logger.info("User logged out")
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error in logout endpoint: {e}")
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
            
            logger.info(f"Crawling website: {start_url}")
            # Create a new event loop for async operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Stream progress updates via SSE
                def generate():
                    async def run_crawl():
                        async for update in crawl_website_progress(start_url):
                            yield f"data: {update}\n\n"
                        # After progress streaming, collect and store data
                        crawled_data = await crawl_website_data(start_url)
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
                        yield f"data: {json.dumps({'status': 'stored', 'items_crawled': len(crawled_data)})}\n\n"
                    
                    loop.run_until_complete(run_crawl())
                
                return Response(generate(), mimetype='text/event-stream')
            finally:
                loop.close()
        
        return render_template('crawl.html')
    except Exception as e:
        logger.error(f"Error in crawl endpoint: {e}")
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
        logger.error(f"Error in search endpoint: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(debug=True)