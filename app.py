from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
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
import subprocess
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
import traceback
from openai import OpenAI
import tenacity
import time
import pdfplumber
import aiohttp
import re
import io
from heapq import heappush, heappop
import threading

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours
socketio = SocketIO(app, async_mode='gevent')
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Initialize crawl state
crawl_state = {'running': False, 'crawled_data': [], 'user_id': None, 'library_id': None, 'browser': None}
stop_event = threading.Event()

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
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'img', 'video', 'audio']):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())[:1000]
    except Exception as e:
        logger.error(f"Error cleaning content: {e}\n{traceback.format_exc()}")
        return ""

async def extract_pdf_text(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
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
        Analyze the provided HTML to identify interactive elements (e.g., buttons, links, dropdowns, tabs, search inputs, accordion toggles, collapsible panels, frames) that could reveal additional links or content when clicked, scrolled, or interacted with. Prioritize elements that toggle expandable sections (e.g., accordion headers with classes like 'collapsible-header', 'pii-tag country-tag', 'accordion', 'toggle', 'directory-item', 'collapse', or attributes like 'aria-expanded', 'data-toggle', 'data-collapse') or trigger dynamic content loading (e.g., 'Load More', search buttons, filter dropdowns, tab switches). Suggest specific actions (e.g., click selectors, scroll, fill input, switch frames) to uncover more links, including multi-step paths (e.g., click an accordion header to expand a section, then extract links). Return a JSON list of actions, each with 'type' ('click', 'scroll', 'fill', 'frame'), 'selector' (CSS selector for click/fill or frame name for frame), 'value' (input value for fill, empty otherwise), and 'priority' (1 for high, 2 for medium, 3 for low). Ensure the response is valid JSON.
        Example response: [
            {"type": "click", "selector": ".collapsible-header, .pii-tag.country-tag", "value": "", "priority": 1},
            {"type": "fill", "selector": "input#search", "value": "query", "priority": 1},
            {"type": "scroll", "selector": "", "value": "", "priority": 2},
            {"type": "frame", "selector": "main_frame", "value": "", "priority": 1}
        ]
        """
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        completion = client.chat.completions.create(
            model="grok-3-latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"HTML: {html[:15000]}"}
            ],
            max_tokens=2000
        )
        raw_response = completion.choices[0].message.content
        logger.debug(f"Grok API raw response: {raw_response}")
        try:
            actions = json.loads(raw_response)
            if not isinstance(actions, list):
                logger.error("Grok API response is not a list")
                return []
            return sorted(actions, key=lambda x: x.get('priority', 3))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Grok API response as JSON: {raw_response}")
            return []
    except Exception as e:
        logger.error(f"Error analyzing page for links: {str(e)}\n{traceback.format_exc()}")
        return []

async def generate_embedding(text, tokenizer, model):
    try:
        process = psutil.Process()
        logger.info(f"Memory usage before embedding: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
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

def normalize_url(url):
    """Normalize URL by adding https:// if missing and ensuring proper format."""
    if not url:
        return None
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            return None
        if not parsed.netloc.startswith('www.'):
            parsed = parsed._replace(netloc='www.' + parsed.netloc)
        return parsed.geturl()
    except ValueError:
        return None

async def FindLinks(page, url, visited_urls, to_visit, depth, crawl_depth, base_domain, domain_restriction, stop_event):
    new_links = []
    try:
        frames = [f for f in page.frames if f.url != 'about:blank' and f.url and not re.search(r'(ads|analytics|doubleclick|googlesyndication)', f.url, re.IGNORECASE)][:5]
        logger.info(f"Found {len(frames)} valid frames on {url}")
        for frame in [page.main_frame] + frames:
            if stop_event.is_set():
                logger.info("Stop event detected, exiting FindLinks")
                break
            frame_name = frame.name or frame.url or 'main'
            logger.debug(f"Processing frame: {frame_name}")
            
            try:
                await frame.evaluate('true')
            except Exception as e:
                logger.debug(f"Skipping detached frame {frame_name}: {e}")
                continue
            
            try:
                for _ in range(2):
                    await frame.evaluate('window.scrollTo(0, document.body.scrollHeight);')
                    await frame.wait_for_timeout(2000)
            except Exception as e:
                logger.debug(f"Error scrolling in frame {frame_name}: {e}")
            
            accordion_selectors = [
                '.collapsible-header', '.pii-tag.country-tag',
                '[aria-expanded="false"]',
                '.accordion, .accordion-toggle, .accordion-header',
                '.expand, .toggle, .collapsible, .collapse',
                '[data-toggle], [data-expand], [data-collapse]',
                '[role="button"], [aria-controls]',
                '.directory-item, .collapse-header, .panel-heading',
            ]
            click_count = 0
            max_clicks = 500
            for selector in accordion_selectors:
                if click_count >= max_clicks or stop_event.is_set():
                    logger.warning(f"Reached max click limit ({max_clicks}) or stop event in frame: {frame_name}")
                    break
                elements = await frame.query_selector_all(selector)
                logger.debug(f"Found {len(elements)} elements for selector: {selector} in frame: {frame_name}")
                for element in elements:
                    if click_count >= max_clicks or stop_event.is_set():
                        break
                    try:
                        is_visible = await element.is_visible()
                        if not is_visible:
                            continue
                        aria_expanded = await element.get_attribute('aria-expanded')
                        if aria_expanded == 'false' or aria_expanded is None or selector in ['.collapsible-header', '.pii-tag.country-tag']:
                            for _ in range(2):
                                try:
                                    await frame.evaluate('''(element) => element.click()''', element)
                                    await frame.wait_for_timeout(3000)
                                    await frame.wait_for_selector('.collapsible-body', state='visible', timeout=5000)
                                    click_count += 1
                                    logger.info(f"Clicked collapsible element #{click_count} with selector: {selector} in frame: {frame_name}")
                                    links = await frame.evaluate('''() => {
                                        return Array.from(document.querySelectorAll('.collapsible-body a[href]')).map(a => a.href);
                                    }''')
                                    for link in links:
                                        absolute_url = urllib.parse.urljoin(url, link)
                                        if (absolute_url not in visited_urls and 
                                            absolute_url not in [u[2] for u in to_visit] and 
                                            absolute_url.startswith(('http://', 'https://'))):
                                            parsed_url = urllib.parse.urlparse(absolute_url)
                                            if domain_restriction and parsed_url.netloc != base_domain:
                                                continue
                                            new_links.append((absolute_url, depth + 1))
                                    break
                                except Exception as e:
                                    logger.debug(f"Retry clicking element {selector} in frame: {e}")
                                    await frame.wait_for_timeout(1000)
                    except Exception as e:
                        logger.debug(f"Error processing element {selector} in frame: {e}")
            
            try:
                actions = await analyze_page_for_links(frame)
                if not actions:
                    actions = [
                        {"type": "click", "selector": ".collapsible-header, .pii-tag.country-tag, [aria-expanded='false'], .accordion-toggle, .collapsible, [data-toggle], .directory-item, a[href*='resource']", "value": "", "priority": 1},
                        {"type": "scroll", "selector": "", "value": "", "priority": 2}
                    ]
                
                for action in actions[:10]:
                    if stop_event.is_set():
                        break
                    if action['type'] == 'click' and action['selector']:
                        elements = await frame.query_selector_all(action['selector'])
                        for element in elements[:3]:
                            try:
                                is_visible = await element.is_visible()
                                if is_visible:
                                    await element.click(timeout=5000)
                                    await frame.wait_for_timeout(3000)
                                    links = await frame.evaluate('''() => {
                                        return Array.from(document.querySelectorAll('a[href]')).map(a => a.href);
                                    }''')
                                    for link in links:
                                        absolute_url = urllib.parse.urljoin(url, link)
                                        if (absolute_url not in visited_urls and 
                                            absolute_url not in [u[2] for u in to_visit] and 
                                            absolute_url.startswith(('http://', 'https://'))):
                                            parsed_url = urllib.parse.urlparse(absolute_url)
                                            if domain_restriction and parsed_url.netloc != base_domain:
                                                continue
                                            new_links.append((absolute_url, depth + 1))
                            except Exception as e:
                                logger.debug(f"Error clicking element {action['selector']} in frame: {e}")
                    elif action['type'] == 'scroll':
                        await frame.evaluate('window.scrollTo(0, document.body.scrollHeight);')
                        await frame.wait_for_timeout(3000)
                    elif action['type'] == 'fill' and action['selector'] and action['value']:
                        input_field = await frame.query_selector(action['selector'])
                        if input_field and await input_field.is_visible():
                            await frame.fill(action['selector'], action['value'])
                            await frame.wait_for_timeout(2000)
                            await frame.keyboard.press("Enter")
                            await frame.wait_for_timeout(3000)
            except Exception as e:
                logger.error(f"Error in Grok API analysis for frame {frame_name}: {str(e)}")
            
            await frame.wait_for_timeout(3000)
            
            try:
                links = await frame.evaluate('''() => {
                    const urls = [];
                    document.querySelectorAll('a[href], [href], button, [role="link"], [onclick], [data-href], [data-nav], [data-url], [data-link], [ng-href], [data-ng-href], [data-action], [data-target], meta[content][http-equiv="refresh"], link[rel="sitemap"]').forEach(el => {
                        let url = el.href || el.getAttribute('href') || el.getAttribute('data-href') || el.getAttribute('data-nav') || el.getAttribute('data-url') || el.getAttribute('data-link') || el.getAttribute('ng-href') || el.getAttribute('data-ng-href') || el.getAttribute('data-action') || el.getAttribute('data-target');
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
                    if (absolute_url not in visited_urls and 
                        absolute_url not in [u[2] for u in to_visit] and 
                        absolute_url.startswith(('http://', 'https://'))):
                        parsed_url = urllib.parse.urlparse(absolute_url)
                        if domain_restriction and parsed_url.netloc != base_domain:
                            continue
                        new_links.append((absolute_url, depth + 1))
            except Exception as e:
                logger.error(f"Error extracting links in frame {frame_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in FindLinks for {url}: {str(e)}")
    
    return new_links

async def ScanLinks(browser, links, max_results, items_crawled, stop_event):
    scanned_links = []
    for url, depth in links:
        if items_crawled >= max_results or stop_event.is_set():
            break
        try:
            page = await browser.new_page()
            try:
                logger.info(f"Scanning URL: {url}")
                response = await page.goto(url, timeout=90000, wait_until='networkidle')
                if response is None or response.status >= 400:
                    logger.warning(f"Failed to load {url}: Status {response.status if response else 'None'}")
                    await page.close()
                    continue
                
                content_type = response.headers.get('content-type', '').lower()
                cleaned_content = None
                if 'text/html' in content_type:
                    content = await page.content()
                    cleaned_content = clean_content(content)
                    if cleaned_content and len(cleaned_content.strip()) >= 100:
                        scanned_links.append((url, cleaned_content, depth))
                        logger.info(f"Meaningful content found for {url}")
                elif 'application/pdf' in content_type:
                    cleaned_content = await extract_pdf_text(url)
                    if cleaned_content and len(cleaned_content.strip()) >= 100:
                        scanned_links.append((url, cleaned_content, depth))
                        logger.info(f"Meaningful PDF content found for {url}")
            except Exception as e:
                logger.error(f"Error scanning {url}: {str(e)}")
            finally:
                await page.close()
        except Exception as e:
            logger.error(f"Error creating page for {url}: {str(e)}")
    
    return scanned_links

async def CrawlLinks(scanned_links, library_id, max_results, items_crawled, stop_event):
    crawled_data = []
    embedding_tokenizer, embedding_model = await load_embedding_model()
    for url, cleaned_content, depth in scanned_links:
        if items_crawled >= max_results or stop_event.is_set():
            break
        try:
            embedding = await generate_embedding(cleaned_content, embedding_tokenizer, embedding_model)
            if embedding is not None:
                crawled_data.append((url, cleaned_content, embedding.tolist(), None, library_id))
                items_crawled += 1
                logger.info(f"Crawled and embedded {url}, items_crawled={items_crawled}")
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
    return crawled_data, items_crawled

async def crawl_website(start_url, user_id, library_id, domain_restriction, crawl_depth, links_to_batch, max_results):
    global crawl_state, stop_event
    crawl_state['running'] = True
    crawl_state['crawled_data'] = []
    crawl_state['user_id'] = user_id
    crawl_state['library_id'] = library_id
    crawl_state['browser'] = None
    stop_event.clear()

    logger.info(f"Starting crawl for {start_url} by user {user_id} in library {library_id}, domain_restriction={domain_restriction}, crawl_depth={crawl_depth}, links_to_batch={links_to_batch}, max_results={max_results}")
    process = psutil.Process()
    logger.info(f"Memory usage before crawl: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    logger.info(f"CPU usage before crawl: {psutil.cpu_percent()}%")
    
    visited_urls = set()
    to_visit = []
    def add_to_visit(url, priority, depth):
        heappush(to_visit, (priority, depth, url))
    
    education_keywords = ['/education', '/learning-center', '/resources']
    def get_url_priority(url):
        for keyword in education_keywords:
            if keyword in url.lower():
                return 0
        return 1
    
    base_domain = urllib.parse.urlparse(start_url).netloc
    add_to_visit(start_url, get_url_priority(start_url), 0)
    links_found = 0
    links_scanned = 0
    items_crawled = 0
    
    conn = sqlite3.connect('progress.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO progress (user_id, url, library_id, links_found, links_scanned, items_crawled, status, current_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (str(user_id), start_url, library_id, links_found, links_scanned, items_crawled, "running", start_url))
    conn.commit()
    
    browser = None
    try:
        async with async_playwright() as p:
            logger.info(f"Memory usage before browser launch: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            logger.info(f"CPU usage before browser launch: {psutil.cpu_percent()}%")
            try:
                browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'])
                crawl_state['browser'] = browser
            except Exception as e:
                logger.error(f"Failed to launch browser: {str(e)}\n{traceback.format_exc()}")
                c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                          (f"error: Failed to launch browser: {str(e)}", "", str(user_id), start_url, library_id))
                conn.commit()
                conn.close()
                crawl_state['running'] = False
                socketio.emit('crawl_update', {
                    'links_found': links_found,
                    'links_scanned': links_scanned,
                    'items_crawled': items_crawled,
                    'status': f"error: {str(e)}"
                }, namespace='/crawl')
                return []
            
            logger.info(f"Memory usage after browser launch: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            logger.info(f"CPU usage after browser launch: {psutil.cpu_percent()}%")
            
            # Process the starting URL first
            try:
                page = await browser.new_page()
                response = await page.goto(start_url, timeout=90000, wait_until='networkidle')
                if response and response.status < 400:
                    try:
                        await page.wait_for_function('window.angular && window.angular.element(document).injector().get("$http").pendingRequests.length === 0', timeout=20000)
                        await page.wait_for_function('typeof $ !== "undefined" && $(".collapsible").collapsible', timeout=20000)
                        await page.wait_for_timeout(5000)
                    except Exception as e:
                        logger.debug(f"Stabilization wait failed: {e}")
                    
                    try:
                        modal_close_selectors = ['.modal-close', '.close', '[aria-label="close"]', 'button.close', '.ot-sdk-close']
                        for selector in modal_close_selectors:
                            close_buttons = await page.query_selector_all(selector)
                            for button in close_buttons:
                                try:
                                    await button.click(timeout=5000)
                                    await page.wait_for_timeout(2000)
                                except Exception as e:
                                    logger.debug(f"Error closing modal {selector}: {e}")
                    except Exception as e:
                        logger.error(f"Error closing modals: {str(e)}")
                    
                    content_type = response.headers.get('content-type', '').lower()
                    cleaned_content = None
                    if 'text/html' in content_type:
                        content = await page.content()
                        cleaned_content = clean_content(content)
                        if cleaned_content and len(cleaned_content.strip()) >= 100 and items_crawled < max_results:
                            embedding = await generate_embedding(cleaned_content, *await load_embedding_model())
                            if embedding is not None:
                                crawl_state['crawled_data'].append((start_url, cleaned_content, embedding.tolist(), None, library_id))
                                items_crawled += 1
                                logger.info(f"Starting URL content crawled, items_crawled={items_crawled}")
                    elif 'application/pdf' in content_type:
                        cleaned_content = await extract_pdf_text(start_url)
                        if cleaned_content and len(cleaned_content.strip()) >= 100 and items_crawled < max_results:
                            embedding = await generate_embedding(cleaned_content, *await load_embedding_model())
                            if embedding is not None:
                                crawl_state['crawled_data'].append((start_url, cleaned_content, embedding.tolist(), None, library_id))
                                items_crawled += 1
                                logger.info(f"Starting URL PDF content crawled, items_crawled={items_crawled}")
                    
                    new_links = await FindLinks(page, start_url, visited_urls, to_visit, 0, crawl_depth, base_domain, domain_restriction, stop_event)
                    for link_url, link_depth in new_links:
                        add_to_visit(link_url, get_url_priority(link_url), link_depth)
                    links_found += len(new_links)
                    visited_urls.add(start_url)
                    
                    c.execute("UPDATE progress SET links_found = ?, items_crawled = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                              (links_found, items_crawled, start_url, str(user_id), start_url, library_id))
                    conn.commit()
                    socketio.emit('crawl_update', {
                        'links_found': links_found,
                        'links_scanned': links_scanned,
                        'items_crawled': items_crawled,
                        'status': 'running'
                    }, namespace='/crawl')
                await page.close()
            except Exception as e:
                logger.error(f"Error processing starting URL {start_url}: {str(e)}")
            
            # Process batches of links
            while items_crawled < max_results and crawl_state['running'] and not stop_event.is_set():
                batch_links = []
                while len(batch_links) < links_to_batch and to_visit and not stop_event.is_set():
                    try:
                        priority, depth, url = heappop(to_visit)
                        if depth > crawl_depth:
                            continue
                        if url in visited_urls or not url.startswith(('http://', 'https://')):
                            continue
                        if re.search(r'\.(jpg|jpeg|png|gif|bmp|svg|ico|mp4|avi|mov|wmv|flv|webm|mp3|wav|ogg|css|js|woff|woff2|ttf|eot)$', url, re.IGNORECASE):
                            continue
                        if 'modal' in url.lower():
                            continue
                        parsed_url = urllib.parse.urlparse(url)
                        if domain_restriction and parsed_url.netloc != base_domain:
                            continue
                        
                        batch_links.append((url, depth))
                        visited_urls.add(url)
                    except Exception as e:
                        logger.error(f"Error selecting link: {str(e)}")
                        continue
                
                if not batch_links or stop_event.is_set():
                    break
                
                links_found += len(batch_links)
                scanned_links = await ScanLinks(browser, batch_links, max_results, items_crawled, stop_event)
                links_scanned += len(batch_links)
                
                new_crawled_data, items_crawled = await CrawlLinks(scanned_links, library_id, max_results, items_crawled, stop_event)
                crawl_state['crawled_data'].extend(new_crawled_data)
                
                c.execute("UPDATE progress SET links_found = ?, links_scanned = ?, items_crawled = ?, status = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                          (links_found, links_scanned, items_crawled, "running", str(user_id), start_url, library_id))
                conn.commit()
                socketio.emit('crawl_update', {
                    'links_found': links_found,
                    'links_scanned': links_scanned,
                    'items_crawled': items_crawled,
                    'status': 'running'
                }, namespace='/crawl')
                
                if items_crawled >= max_results or stop_event.is_set():
                    break
            
            logger.info(f"Crawl complete: items_crawled={items_crawled}, links_found={links_found}, links_scanned={links_scanned}")
            c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                      ("complete", "", str(user_id), start_url, library_id))
            conn.commit()
            socketio.emit('crawl_update', {
                'links_found': links_found,
                'links_scanned': links_scanned,
                'items_crawled': items_crawled,
                'status': 'complete'
            }, namespace='/crawl')
    except Exception as e:
        logger.error(f"Unexpected error during crawl: {str(e)}\n{traceback.format_exc()}")
        c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND url = ? AND library_id = ?",
                  (f"error: {str(e)}", "", str(user_id), start_url, library_id))
        conn.commit()
        socketio.emit('crawl_update', {
            'links_found': links_found,
            'links_scanned': links_scanned,
            'items_crawled': items_crawled,
            'status': f"error: {str(e)}"
        }, namespace='/crawl')
    finally:
        if browser:
            try:
                await browser.close()
                logger.info("Playwright browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")
        crawl_state['browser'] = None
        conn.close()
        await unload_embedding_model()
        crawl_state['running'] = False
        if not crawl_state['crawled_data'] and links_found == links_scanned:
            flash("No new textual content found; all URLs already exist or lack meaningful text.", 'info')
        return crawl_state['crawled_data']

@app.route('/stop_crawl', methods=['POST'])
@login_required
def stop_crawl():
    global crawl_state, stop_event
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user attempting to stop crawl")
            return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
        
        if not crawl_state['running']:
            logger.info("No crawl is running to stop")
            return jsonify({'status': 'error', 'message': 'No crawl is running'}), 400
        
        if crawl_state['user_id'] != current_user.id:
            logger.error("User not authorized to stop this crawl")
            return jsonify({'status': 'error', 'message': 'Not authorized to stop this crawl'}), 403
        
        logger.info(f"Stopping crawl for user {current_user.id}, library {crawl_state['library_id']}")
        commit = request.form.get('commit', 'false').lower() == 'true'
        stop_event.set()
        crawl_state['running'] = False
        
        # Close Playwright browser
        if crawl_state['browser']:
            try:
                await crawl_state['browser'].close()
                logger.info("Playwright browser closed via stop_crawl")
                crawl_state['browser'] = None
            except Exception as e:
                logger.error(f"Error closing browser in stop_crawl: {str(e)}")
        
        if commit and crawl_state['crawled_data']:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                query = """
                INSERT INTO documents (url, content, embedding, file_path, library_id)
                VALUES %s
                ON CONFLICT (url, library_id) DO NOTHING
                """
                execute_values(cur, query, crawl_state['crawled_data'])
                conn.commit()
                cur.close()
                conn.close()
                flash(f"Stored {len(crawl_state['crawled_data'])} items in library.", 'success')
                logger.info(f"Committed {len(crawl_state['crawled_data'])} items to library")
            except Exception as e:
                logger.error(f"Error committing crawl data: {str(e)}")
                flash(f"Error storing crawl data: {str(e)}", 'error')
        
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        c.execute("UPDATE progress SET status = ?, current_url = ? WHERE user_id = ? AND library_id = ?",
                  ("stopped", "", str(current_user.id), crawl_state['library_id']))
        conn.commit()
        conn.close()
        
        socketio.emit('crawl_update', {
            'links_found': 0,
            'links_scanned': 0,
            'items_crawled': 0,
            'status': 'stopped'
        }, namespace='/crawl')
        logger.info("Sent stopped status via Socket.IO")
        
        return jsonify({'status': 'success', 'message': 'Crawl stopped' + (' and data committed' if commit else '')})
    except Exception as e:
        logger.error(f"Error stopping crawl: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
def logout():
    try:
        logout_user()
        logger.info("User logged out")
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error in logout endpoint: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500

@app.route('/libraries', methods=['GET', 'POST'])
@login_required
def libraries():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing libraries endpoint")
            flash('Please log in to access libraries.', 'error')
            return redirect(url_for('login'))
        
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
def view_library(library_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing view_library endpoint")
            flash('Please log in to view libraries.', 'error')
            return redirect(url_for('login'))
        
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
def delete_library_content(content_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing delete_library_content endpoint")
            flash('Please log in to delete content.', 'error')
            return redirect(url_for('login'))
        
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
def delete_library(library_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing delete_library endpoint")
            flash('Please log in to delete libraries.', 'error')
            return redirect(url_for('libraries'))
        
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
def add_library():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing add_library endpoint")
            return jsonify({'status': 'error', 'message': 'Please log in to add a library'}), 401
        
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
def prompts():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing prompts endpoint")
            flash('Please log in to access prompts.', 'error')
            return redirect(url_for('login'))
        
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
def edit_prompt(prompt_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing edit_prompt endpoint")
            flash('Please log in to edit prompts.', 'error')
            return redirect(url_for('prompts'))
        
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
def delete_prompt(prompt_id):
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing delete_prompt endpoint")
            flash('Please log in to delete prompts.', 'error')
            return redirect(url_for('prompts'))
        
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
def crawl():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing crawl endpoint")
            flash('Please log in to start a crawl.', 'error')
            return redirect(url_for('login'))
        
        logger.debug(f"User authenticated: ID {current_user.id}, email {current_user.email}")
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM libraries WHERE user_id = %s", (int(current_user.id),))
        libraries = cur.fetchall()
        cur.close()
        conn.close()
        
        if request.method == 'POST':
            start_url = normalize_url(request.form.get('url'))
            library_id = request.form.get('library_id')
            domain_restriction = request.form.get('domain_restriction') == 'yes'
            try:
                crawl_depth = int(request.form.get('crawl_depth'))
                links_to_batch = int(request.form.get('links_to_batch'))
                max_results = int(request.form.get('max_results'))
            except ValueError:
                flash('Crawl Depth, Links to Batch, and Max Number of Results must be valid numbers.', 'error')
                return render_template('crawl.html', libraries=libraries)
            
            if not start_url or not library_id:
                flash('URL and library selection cannot be empty.', 'error')
                return render_template('crawl.html', libraries=libraries)
            if links_to_batch < 1 or max_results < 1 or crawl_depth < 1:
                flash('Crawl Depth, Links to Batch, and Max Number of Results must be positive.', 'error')
                return render_template('crawl.html', libraries=libraries)
            if links_to_batch > max_results:
                flash('Links to Batch must be less than or equal to Max Number of Results.', 'error')
                return render_template('crawl.html', libraries=libraries)
            
            logger.info(f"Starting crawl for {start_url} by user {current_user.id} in library {library_id}")
            try:
                crawled_data = asyncio.run(crawl_website(start_url, current_user.id, int(library_id), domain_restriction, crawl_depth, links_to_batch, max_results))
                
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
            except Exception as e:
                logger.error(f"Error during crawl: {str(e)}\n{traceback.format_exc()}")
                flash(f"Error during crawl: {str(e)}", 'error')
                return render_template('crawl.html', libraries=libraries)
        
        return render_template('crawl.html', libraries=libraries)
    except Exception as e:
        logger.error(f"Error in crawl endpoint: {e}\n{traceback.format_exc()}")
        flash(f"Error: {str(e)}", 'error')
        return render_template('crawl.html', libraries=[])

@app.route('/crawl_progress', methods=['GET'])
@login_required
def crawl_progress():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user in crawl_progress endpoint")
            return jsonify({"status": "error", "message": "User not authenticated"}), 401
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        c.execute("SELECT links_found, links_scanned, items_crawled, status FROM progress WHERE user_id = ? AND url = ? AND library_id = ? ORDER BY rowid DESC LIMIT 1",
                  (str(current_user.id), request.args.get('url'), request.args.get('library_id')))
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
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user in current_url endpoint")
            return jsonify({"status": "error", "message": "User not authenticated"}), 401
        conn = sqlite3.connect('progress.db')
        c = conn.cursor()
        c.execute("SELECT current_url FROM progress WHERE user_id = ? AND url = ? AND library_id = ? ORDER BY rowid DESC LIMIT 1",
                  (str(current_user.id), request.args.get('url'), request.args.get('library_id')))
        result = c.fetchone()
        conn.close()
        return jsonify({"current_url": result[0] if result and result[0] else ""})
    except Exception as e:
        logger.error(f"Error in current_url: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    try:
        if not current_user.is_authenticated:
            logger.error("Unauthenticated user accessing search endpoint")
            flash('Please log in to perform a search.', 'error')
            return redirect(url_for('login'))
        
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
            try:
                tokenizer, model = asyncio.run(load_embedding_model())
                query_embedding = asyncio.run(generate_embedding(query, tokenizer, model))
                asyncio.run(unload_embedding_model())
            except Exception as e:
                logger.error(f"Error generating query embedding: {str(e)}")
                flash('Failed to generate query embedding.', 'error')
                return render_template('search.html', libraries=libraries, prompts=prompts)
            
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
def test_playwright():
    try:
        logger.info("Starting Playwright test")
        async def run_playwright():
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto('https://example.com')
                content = await page.content()
                await browser.close()
                return content
        
        content = asyncio.run(run_playwright())
        logger.info("Playwright test completed successfully")
        return f"Playwright test successful: {len(content)} bytes"
    except Exception as e:
        logger.error(f"Playwright test failed: {str(e)}\n{traceback.format_exc()}")
        return f"Playwright test failed: {str(e)}"

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    socketio.run(app, debug=True)