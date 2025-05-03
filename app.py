from flask import Flask, render_template, request, redirect, url_for, flash
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
from crawl4ai import WebCrawler
import urllib.parse
import re

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MobileBERT model and tokenizer lazily
tokenizer = None
model = None

def load_mobilebert_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading MobileBERT model...")
        tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
        model = AutoModel.from_pretrained('google/mobilebert-uncased')
        logger.info("MobileBERT model loaded.")
    return tokenizer, model

# Database connection function
def get_db_connection():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    return conn

class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user:
        return User(user[0], user[1])
    return None

def generate_embedding(text):
    """Generate embedding for a single text."""
    tokenizer, model = load_mobilebert_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def crawl_website(start_url):
    """Crawl a website and extract text and PDFs."""
    crawler = WebCrawler()
    crawler.warmup()
    
    crawled_data = []
    visited_urls = set()
    to_visit = [start_url]
    base_domain = urllib.parse.urlparse(start_url).netloc
    
    while to_visit and len(crawled_data) < 100:  # Limit for testing
        url = to_visit.pop(0)
        if url in visited_urls:
            continue
        visited_urls.add(url)
        
        try:
            result = crawler.run(url=url, follow_links=True, max_depth=2)
            if result.success:
                content = result.markdown
                if content:
                    embedding = generate_embedding(content)
                    crawled_data.append((url, content, embedding.tolist(), None))
                
                # Extract PDFs
                for link in result.links:
                    if link.endswith('.pdf'):
                        pdf_path = f"data/pdfs/{urllib.parse.quote(link, safe='')}.pdf"
                        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
                        with urlopen(link) as response, open(pdf_path, 'wb') as out_file:
                            out_file.write(response.read())
                        pdf_text = extract_pdf_text(pdf_path)  # Placeholder for PDF extraction
                        if pdf_text:
                            embedding = generate_embedding(pdf_text)
                            crawled_data.append((link, pdf_text, embedding.tolist(), pdf_path))
                
                # Add new links to visit (same domain)
                for link in result.links:
                    if urllib.parse.urlparse(link).netloc == base_domain and link not in visited_urls:
                        to_visit.append(link)
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
    
    return crawled_data

def extract_pdf_text(pdf_path):
    """Placeholder for PDF text extraction."""
    return ""  # Implement using pdfplumber or PyPDF2 later

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, email, password_hash FROM users WHERE id = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user and check_password_hash(user[2], password):
            login_user(User(user[0], user[1]))
            return redirect(url_for('index'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/crawl', methods=['GET', 'POST'])
@login_required
def crawl():
    if request.method == 'POST':
        start_url = request.form.get('url')
        if not start_url:
            flash('URL cannot be empty.', 'error')
            return render_template('crawl.html')
        
        logger.info(f"Crawling website: {start_url}")
        crawled_data = crawl_website(start_url)
        
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
        
        flash(f"Crawled {len(crawled_data)} items.", 'success')
        return redirect(url_for('search'))
    
    return render_template('crawl.html')

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            flash('Query cannot be empty.', 'error')
            return render_template('search.html')
        
        logger.info(f"Search query: {query}")
        query_embedding = generate_embedding(query)
        
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

if __name__ == '__main__':
    app.run(debug=True)