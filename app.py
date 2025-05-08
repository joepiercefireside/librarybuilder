from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import psycopg2
from psycopg2.extras import execute_values
from werkzeug.security import generate_password_hash, check_password_hash
import os
import logging
import asyncio
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
import urllib.parse
import sqlite3
import json
import psutil
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import traceback
from openai import OpenAI
import tenacity
import time
import pdfplumber
import aiohttp
import re

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
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
                 (user_id TEXT, url TEXT, library_id INTEGER, links_found INTEGER, links_scanned INTEGER, items_crawled INTEGER, status TEXT, current_url TEXT)''')
    conn.commit()
    conn.close()

init_progress_db()

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
async def load_embedding_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading all-MiniLM-L6-v2 model...")
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
            logger.info("all-MiniLM-L6-v2 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}\n{traceback.format_exc()}")
            # Fallback to basic HTTP download without hf_xet
            try:
                os.environ["HF_HUB_DISABLE_XET"] = "1"
                tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
                model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/tmp/hf_cache')
                logger.info("Fallback: all-MiniLM-L6-v2 model loaded without hf_xet.")
            except Exception as e2:
                logger.error(f"Fallback failed: {str(e2)}\n{traceback.format_exc()}")
                raise
    return tokenizer, model

async def unload_embedding_model():
    global tokenizer, model
    tokenizer = None
    model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info("Unloaded all-MiniLM-L6-v2 model to free memory.")

def get_db_connection():
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        logger.info("Database connection established.")
        return conn
    except KeyError as e:
        logger.error(f"Missing DATABASE_URL: {e}")
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
        cur.execute("SELECT id, email FROM users WHERE id = %s", (int(user_id),))
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
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())[:1000]
    except Exception as e:
        logger.error(f"Error cleaning content: {e}\n{traceback.format_exc()}")
        return html[:1000]

async def extract_pdf_text(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200 or 'application/pdf' not in response.headers.get('Content-Type', ''):
                    return None
                pdf_data = await response.read()
                with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                    text = ''
                    for page in pdf.pages:
                        text += page.extract_text() or ''
                    return ' '.join(text.split())[:1000] if text else None
    except Exception as e:
        logger.error(f"Error extracting PDF text from {url}: {e}\n{traceback.format_exc()}")
        return None

async def analyze_page_for_links(page):
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.error("XAI_API_KEY not set")
            return []
        
        html = await page.content()
        prompt = """
        Analyze the provided HTML to identify potential interactive elements (e.g., buttons, links, dynamic lists, endless scroll triggers) that could reveal additional pages when clicked or scrolled. Suggest specific actions (e.g., click selectors, scroll) to uncover more links. Return a JSON list of actions, each with 'type' ('click' or 'scroll') and 'selector' (CSS selector for click or empty for scroll).
        """
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        completion = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"HTML: {html[:2000]}"}
            ],
            max_tokens=500
        )
        actions = json.loads(completion.choices[0].message.content) if completion.choices[0].message.content else []
        return actions
    except Exception as e:
        logger.error(f"Error analyzing page for links: {e}\n{traceback.format_exc()}")
        return []

async def generate_embedding(text):
    try:
        process = psutil.Process()
        logger.info(f"Memory usage before embedding: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        tokenizer, model = await load_embedding_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        await unload_embedding_model()
        logger.info(f"Memory usage after embedding: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}\n{traceback.format_exc()}")
        return None

def query_grok_api(query, context, prompt):
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.error("XAI_API_KEY not set")
            return "Error: xAI API key not configured"
        
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        completion = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Based on the following context, answer the question: {query}\n\nContext: {context}"}
            ],
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying xAI Grok API: {str(e)}\n{traceback.format_exc()}")
        return f"Fallback: Unable to generate AI summary. Please check API key or endpoint."

async def crawl_website(start_url, user_id, library_id):
    logger.info(f"Starting crawl for {start_url} by user {user_id} in library {library_id}")
    process = psutil.Process()
    logger.info(f"Memory usage before crawl: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    visited_urls = set()
    to_visit = [start_url]
    base_domain = urllib.parse.urlparse(start_url).netloc
    links_found = 1
    links_scanned = 0
    items_crawled = 0
    max_items = 20
    crawled_data = []
    
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO progress (user_id, url, library_id, links_found, links_scanned, items_crawled, status, current_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (user_id, start_url, library_id, links_found, links_scanned, items_crawled, "running", start_url))
    conn.commit()
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'])
            try:
                while to_visit and items_crawled < max_items:
                    url = to_visit.pop(0)
                    if url in visited_urls or not url.startswith(('http://', 'https://')):
                        logger.debug(f"Skipping invalid or visited URL: {url}")
                        continue
                    # Skip non-textual content
                    if re.search(r'\.(jpg|jpeg|png|gif|bmp|svg|ico|mp4|avi|mov|wmv|flv|webm|mp3|wav|ogg)$', url, re.IGNORECASE):
                        logger.debug(f"Skipping non-textual URL: {url}")
                        continue
                    visited_urls.add(url)
                    links_scanned += 1
                    logger.debug(f"Scanning URL: {url}")
                    
                    # Update current_url in progress
                    c.execute("UPDATE progress SET current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                              (url, user_id, start_url, library_id))
                    conn.commit()
                    
                    try:
                        page = await browser.new_page()
                        response = await page.goto(url, timeout=30000, wait_until='networkidle')
                        if response is None or response.status >= 400:
                            logger.warning(f"Failed to load {url}: Status {response.status if response else 'None'}")
                            await page.close()
                            continue
                        
                        # Simulate scrolling
                        await page.evaluate('''() => {
                            window.scrollTo(0, document.body.scrollHeight);
                        }''')
                        await page.wait_for_timeout(2000)
                        
                        # Analyze page for dynamic links
                        actions = await analyze_page_for_links(page)
                        for action in actions:
                            try:
                                if action['type'] == 'click' and action['selector']:
                                    await page.click(action['selector'], timeout=5000)
                                    await page.wait_for_timeout(2000)
                                elif action['type'] == 'scroll':
                                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight);')
                                    await page.wait_for_timeout(2000)
                            except Exception as e:
                                logger.debug(f"Error performing action {action}: {e}")
                        
                        content_type = response.headers.get('content-type', '').lower()
                        content = None
                        if 'text/html' in content_type:
                            content = await page.content()
                            cleaned_content = clean_content(content)
                            if not cleaned_content or len(cleaned_content.strip()) < 100:
                                logger.warning(f"No valid textual content found for {url}")
                                await page.close()
                                continue
                        elif 'application/pdf' in content_type:
                            cleaned_content = await extract_pdf_text(url)
                            if not cleaned_content or len(cleaned_content.strip()) < 100:
                                logger.warning(f"No valid textual content in PDF at {url}")
                                await page.close()
                                continue
                        else:
                            logger.debug(f"Skipping unsupported content type {content_type} for {url}")
                            await page.close()
                            continue
                        
                        embedding = await generate_embedding(cleaned_content)
                        if embedding is None:
                            logger.warning(f"Failed to generate embedding for {url}")
                            await page.close()
                            continue
                        
                        items_crawled += 1
                        crawled_data.append((url, cleaned_content, embedding.tolist(), None, library_id))
                        logger.debug(f"Content found for {url}, items_crawled={items_crawled}")
                        
                        links = await page.evaluate('''() => {
                            const urls = [];
                            document.querySelectorAll('a[href], button, [role="link"], [onclick], [data-href], [data-nav], [data-url], [data-link], meta[content][http-equiv="refresh"], link[rel="sitemap"]').forEach(el => {
                                let url = el.href || el.getAttribute('data-href') || el.getAttribute('data-nav') || el.getAttribute('data-url') || el.getAttribute('data-link');
                                if (!url && el.getAttribute('onclick')) {
                                    const match = el.getAttribute('onclick').match(/(?:location\.href|window\.open|navigateTo|window\.location\.assign|window\.location\.replace)\(['"]([^'"]+)['"]/);
                                    if (match) url = match[1];
                                }
                                if (!url && el.tagName === 'META' && el.getAttribute('http-equiv') === 'refresh') {
                                    const content = el.getAttribute('content');
                                    const match = content.match(/url=(.+)$/i);
                                    if (match) url = match[1];
                                }
                                if (!url && el.tagName === 'LINK' && el.getAttribute('rel') === 'sitemap') {
                                    url = el.href;
                                }
                                if (url) urls.push(url);
                            });
                            const scripts = document.querySelectorAll('script');
                            scripts.forEach(script => {
                                const text = script.textContent || script.src;
                                const matches = text.match(/(?:location\.href|window\.location|navigateTo|open)\(['"]([^'"]+)['"]/g);
                                if (matches) {
                                    matches.forEach(match => {
                                        const urlMatch = match.match(/['"]([^'"]+)['"]/);
                                        if (urlMatch) urls.push(urlMatch[1]);
                                    });
                                }
                            });
                            return [...new Set(urls)];
                        }''')
                        for link in links:
                            absolute_url = urllib.parse.urljoin(url, link)
                            parsed_url = urllib.parse.urlparse(absolute_url)
                            if parsed_url.netloc == base_domain and absolute_url not in visited_urls and absolute_url not in to_visit and parsed_url.scheme in ('http', 'https'):
                                to_visit.append(absolute_url)
                                links_found += 1
                                logger.debug(f"New link found: {absolute_url}, links_found={links_found}")
                        
                        c.execute("UPDATE progress SET links_found = ?, links_scanned = ?, items_crawled = ?, status = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                                  (links_found, links_scanned, items_crawled, "running", user_id, start_url, library_id))
                        conn.commit()
                        
                        await page.close()
                        logger.info(f"Memory usage after scanning {url}: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                    except Exception as e:
                        logger.error(f"Error crawling {url}: {str(e)}\n{traceback.format_exc()}")
                        c.execute("UPDATE progress SET status = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                                  (f"error: {str(e)}", user_id, start_url, library_id))
                        conn.commit()
                        conn.close()
                        return crawled_data
                logger.info(f"Crawl complete: items_crawled={items_crawled}")
                c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                          ("complete", "", user_id, start_url, library_id))
                conn.commit()
            finally:
                await browser.close()
                conn.close()
        if not crawled_data and links_found == links_scanned:
            flash("No new textual content found; all URLs already exist or lack meaningful text.", 'info')
        return crawled_data
    except Exception as e:
        logger.error(f"Unexpected error in crawl_website: {str(e)}\n{traceback.format_exc()}")
        c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                  (f"error: {str(e)}", "", user_id, start_url, library_id))
        conn.commit()
        conn.close()
        return crawled_data

@app.route('/')
async def index():
    try:
        logger.info("Accessing index page")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/register', methods=['GET', 'POST'])
async def register():
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
async def login():
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
                session.permanent = True
                logger.info(f"User {email} logged in successfully")
                return redirect(url_for('index'))
            flash('Invalid email or password.', 'error')
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error in login endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/logout')
@login_required
async def logout():
    try:
        logout_user()
        logger.info("User logged out")
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error in logout endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/libraries', methods=['GET', 'POST'])
@login_required
async def libraries():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if request.method == 'POST':
            name = request.form.get('name')
            if not name:
                flash('Library name cannot be empty.', 'error')
            else:
                cur.execute("INSERT INTO libraries (user_id, name) VALUES (%s, %s)", (int(current_user.id), name))
                conn.commit()
                flash('Library created successfully.', 'success')
                return redirect(url_for('libraries'))
        
        cur.execute("SELECT id, name FROM libraries WHERE user_id = %s", (int(current_user.id),))
        libraries = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('libraries.html', libraries=libraries)
    except Exception as e:
        logger.error(f"Error in libraries endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return render_template('libraries.html', libraries=[])

@app.route('/libraries/view/<int:library_id>')
@login_required
async def view_library(library_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM libraries WHERE id = %s AND user_id = %s", (library_id, int(current_user.id)))
        library = cur.fetchone()
        if not library:
            flash('Library not found.', 'error')
            return redirect(url_for('libraries'))
        
        cur.execute("SELECT id, url, content FROM documents WHERE library_id = %s", (library_id,))
        contents = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('library_view.html', library=library, contents=contents)
    except Exception as e:
        logger.error(f"Error in view_library endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('libraries'))

@app.route('/libraries/delete_content/<int:content_id>')
@login_required
async def delete_library_content(content_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT library_id FROM documents WHERE id = %s", (content_id,))
        library_id = cur.fetchone()
        if not library_id:
            flash('Content not found.', 'error')
            return redirect(url_for('libraries'))
        
        cur.execute("DELETE FROM documents WHERE id = %s AND library_id IN (SELECT id FROM libraries WHERE user_id = %s)", (content_id, int(current_user.id)))
        conn.commit()
        cur.close()
        conn.close()
        flash('Content deleted successfully.', 'success')
        return redirect(url_for('view_library', library_id=library_id[0]))
    except Exception as e:
        logger.error(f"Error in delete_library_content endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('libraries'))

@app.route('/libraries/delete/<int:library_id>')
@login_required
async def delete_library(library_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM libraries WHERE id = %s AND user_id = %s", (library_id, int(current_user.id)))
        if not cur.fetchone():
            flash('Library not found.', 'error')
            return redirect(url_for('libraries'))
        
        cur.execute("DELETE FROM documents WHERE library_id = %s", (library_id,))
        cur.execute("DELETE FROM libraries WHERE id = %s AND user_id = %s", (library_id, int(current_user.id)))
        conn.commit()
        cur.close()
        conn.close()
        flash('Library and its contents deleted successfully.', 'success')
        return redirect(url_for('libraries'))
    except Exception as e:
        logger.error(f"Error in delete_library endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('libraries'))

@app.route('/add_library', methods=['POST'])
@login_required
async def add_library():
    try:
        name = request.form.get('name')
        if not name:
            flash('Library name cannot be empty.', 'error')
            return jsonify({'status': 'error', 'message': 'Library name cannot be empty'}), 400
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO libraries (user_id, name) VALUES (%s, %s) RETURNING id", (int(current_user.id), name))
        library_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success', 'library_id': library_id, 'library_name': name})
    except Exception as e:
        logger.error(f"Error in add_library endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/prompts', methods=['GET', 'POST'])
@login_required
async def prompts():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if request.method == 'POST':
            name = request.form.get('name')
            content = request.form.get('content')
            if not name or not content:
                flash('Prompt name and content cannot be empty.', 'error')
            else:
                cur.execute("INSERT INTO prompts (user_id, name, content) VALUES (%s, %s, %s)", (int(current_user.id), name, content))
                conn.commit()
                flash('Prompt created successfully.', 'success')
                return redirect(url_for('prompts'))
        
        cur.execute("SELECT id, name, content FROM prompts WHERE user_id = %s", (int(current_user.id),))
        prompts = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('prompts.html', prompts=prompts)
    except Exception as e:
        logger.error(f"Error in prompts endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return render_template('prompts.html', prompts=[])

@app.route('/prompts/edit/<int:prompt_id>', methods=['GET', 'POST'])
@login_required
async def edit_prompt(prompt_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, content FROM prompts WHERE id = %s AND user_id = %s", (prompt_id, int(current_user.id)))
        prompt = cur.fetchone()
        if not prompt:
            flash('Prompt not found.', 'error')
            return redirect(url_for('prompts'))
        
        if request.method == 'POST':
            name = request.form.get('name')
            content = request.form.get('content')
            if not name or not content:
                flash('Prompt name and content cannot be empty.', 'error')
            else:
                cur.execute("UPDATE prompts SET name = %s, content = %s WHERE id = %s AND user_id = %s",
                            (name, content, prompt_id, int(current_user.id)))
                conn.commit()
                flash('Prompt updated successfully.', 'success')
                return redirect(url_for('prompts'))
        
        cur.close()
        conn.close()
        return render_template('prompt_edit.html', prompt=prompt)
    except Exception as e:
        logger.error(f"Error in edit_prompt endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('prompts'))

@app.route('/prompts/delete/<int:prompt_id>')
@login_required
async def delete_prompt(prompt_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM prompts WHERE id = %s AND user_id = %s", (prompt_id, int(current_user.id)))
        conn.commit()
        cur.close()
        conn.close()
        flash('Prompt deleted successfully.', 'success')
        return redirect(url_for('prompts'))
    except Exception as e:
        logger.error(f"Error in delete_prompt endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('prompts'))

@app.route('/crawl', methods=['GET', 'POST'])
@login_required
async def crawl():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM libraries WHERE user_id = %s", (int(current_user.id),))
        libraries = cur.fetchall()
        cur.close()
        conn.close()
        
        if request.method == 'POST':
            start_url = normalize_url(request.form.get('url'))
            library_id = request.form.get('library_id')
            if not start_url or not library_id:
                flash('URL and library selection cannot be empty.', 'error')
                return render_template('crawl.html', libraries=libraries)
            
            logger.info(f"Starting crawl for {start_url} by user {current_user.id} in library {library_id}")
            try:
                crawled_data = await crawl_website(start_url, current_user.id, int(library_id))
                
                if crawled_data:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    query = """
                    INSERT INTO documents (url, content, embedding, file_path, library_id)
                    VALUES %s
                    ON CONFLICT (url, library_id) DO NOTHING
                    """
                    execute_values(cur, query, crawled_data)
                    conn.commit()
                    cur.close()
                    conn.close()
                    flash(f"Stored {len(crawled_data)} items in library.", 'success')
                else:
                    flash("No new textual content found; all URLs already exist or lack meaningful text.", 'info')
                
                return redirect(url_for('crawl', url=start_url, library_id=library_id))
            except psycopg2.errors.UniqueViolation as e:
                logger.info(f"Duplicate URL detected: {str(e)}")
                flash("All pages from that link have already been added to the library.", 'info')
                return redirect(url_for('crawl'))
        
        return render_template('crawl.html', libraries=libraries)
    except Exception as e:
        logger.error(f"Error in crawl endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error during crawl: {str(e)}", 'error')
        return render_template('crawl.html', libraries=[])

@app.route('/crawl_progress', methods=['GET'])
@login_required
def crawl_progress():
    try:
        if not current_user or not current_user.is_authenticated:
            logger.error("Unauthenticated user in crawl_progress: current_user=%s" % current_user)
            return jsonify({"status": "error", "message": "User not authenticated"}), 401
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        c.execute("SELECT links_found, links_scanned, items_crawled, status FROM progress WHERE user_id = ? AND url = ? AND library_id = ? ORDER BY rowid DESC LIMIT 1",
                  (current_user.id, request.args.get('url'), request.args.get('library_id')))
        result = c.fetchone()
        conn.close()
        data = {
            "links_found": result[0] if result else 0,
            "links_scanned": result[1] if result else 0,
            "items_crawled": result[2] if result else 0,
            "status": result[3] if result else "waiting"
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in crawl_progress: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/current_url', methods=['GET'])
@login_required
def current_url():
    try:
        if not current_user or not current_user.is_authenticated:
            logger.error("Unauthenticated user in current_url: current_user=%s" % current_user)
            return jsonify({"status": "error", "message": "User not authenticated"}), 401
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        c.execute("SELECT current_url FROM progress WHERE user_id = ? AND url = ? AND library_id = ? ORDER BY rowid DESC LIMIT 1",
                  (current_user.id, request.args.get('url'), request.args.get('library_id')))
        result = c.fetchone()
        conn.close()
        return jsonify({"current_url": result[0] if result and result[0] else ""})
    except Exception as e:
        logger.error(f"Error in current_url: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search', methods=['GET', 'POST'])
@login_required
async def search():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM libraries WHERE user_id = %s", (int(current_user.id),))
        libraries = cur.fetchall()
        cur.execute("SELECT id, name, content FROM prompts WHERE user_id = %s", (int(current_user.id),))
        prompts = cur.fetchall()
        cur.close()
        conn.close()
        
        if request.method == 'POST':
            query = request.form.get('query')
            library_id = request.form.get('library_id')
            prompt_id = request.form.get('prompt_id')
            if not query or not library_id or not prompt_id:
                flash('Query, library, and prompt selection cannot be empty.', 'error')
                return render_template('search.html', libraries=libraries, prompts=prompts)
            
            logger.info(f"Search query: {query} in library {library_id}")
            query_embedding = await generate_embedding(query)
            if query_embedding is None:
                flash('Failed to generate query embedding.', 'error')
                return render_template('search.html', libraries=libraries, prompts=prompts)
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
            SELECT id, url, content, file_path, embedding
            FROM documents
            WHERE library_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT 5
            """, (int(library_id), query_embedding.tolist()))
            results = cur.fetchall()
            
            cur.execute("SELECT content FROM prompts WHERE id = %s AND user_id = %s", (int(prompt_id), int(current_user.id)))
            prompt = cur.fetchone()
            cur.close()
            conn.close()
            
            if not prompt:
                flash('Selected prompt not found.', 'error')
                return render_template('search.html', libraries=libraries, prompts=prompts)
            
            context = "\n\n".join([result[2] for result in results])
            prompt_answer = query_grok_api(query, context, prompt[0])
            if prompt_answer.startswith("Error") or prompt_answer.startswith("Fallback"):
                prompt_answer = f"{prompt_answer}\n\nRelevant Documents:"
            
            documents = [
                {
                    "url": result[1] or result[3],
                    "snippet": result[2][:100] + ("..." if len(result[2]) > 100 else "")
                }
                for result in results
            ]
            
            if not results:
                documents = [{"url": "", "snippet": "No relevant content found."}]
            
            return render_template('search.html', libraries=libraries, prompts=prompts, query=query, prompt_answer=prompt_answer, documents=documents)
        
        return render_template('search.html', libraries=libraries, prompts=prompts)
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

@app.route('/health')
async def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)