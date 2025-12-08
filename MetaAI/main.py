import os
import logging
from logging.handlers import RotatingFileHandler
import psycopg2
from psycopg2 import sql
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from datetime import datetime, timedelta
import uuid

# Constants
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
WIKI_API_URL = 'https://en.wikipedia.org/w/api.php'
TOPICS = [
    'Emerging technologies',
    'Political shifts',
    'Corporate M&A timelines',
    'Industry trends',
    'Nobel Prizes',
    'Scientific breakthroughs'
]

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('world_report.log', maxBytes=1000000, backupCount=5)
logger.addHandler(handler)

# Database functions
def create_schema():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        cur.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS world_report;'))
        cur.execute(sql.SQL('''
            CREATE TABLE IF NOT EXISTS world_report.scraped_items (
                id UUID PRIMARY KEY,
                topic TEXT,
                title TEXT,
                summary TEXT,
                source_url TEXT,
                scraped_at TIMESTAMP,
                tokens TEXT[]
            );
        '''))
        conn.commit()
        logger.info('Schema created')
    except psycopg2.Error as e:
        logger.error(f'Error creating schema: {e}')

def upsert_item(item):
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        cur.execute(sql.SQL('''
            INSERT INTO world_report.scraped_items (id, topic, title, summary, source_url, scraped_at, tokens)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                topic = EXCLUDED.topic,
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                source_url = EXCLUDED.source_url,
                scraped_at = EXCLUDED.scraped_at,
                tokens = EXCLUDED.tokens;
        '''), (
            item['id'],
            item['topic'],
            item['title'],
            item['summary'],
            item['source_url'],
            item['scraped_at'],
            item['tokens']
        ))
        conn.commit()
        logger.info(f'Item {item["id"]} upserted')
    except psycopg2.Error as e:
        logger.error(f'Error upserting item: {e}')

# Wikipedia scraping utilities
def get_wiki_api_response(topic):
    try:
        response = requests.get(WIKI_API_URL, params={
            'action': 'query',
            'format': 'json',
            'titles': topic,
            'prop': 'extracts',
            'exintro': True
        })
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f'Error getting Wikipedia API response: {e}')
        return None

def get_wiki_html(topic):
    try:
        response = requests.get(f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}')
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f'Error getting Wikipedia HTML: {e}')
        return None

def extract_summary(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Extract the first paragraph
    summary = soup.find('p').text
    return summary

def tokenize_text(text):
    try:
        tokens = word_tokenize(text)
        return tokens
    except Exception as e:
        logger.error(f'Error tokenizing text: {e}')
        return []

# PDF builder
def generate_pdf(items):
    try:
        doc = SimpleDocTemplate('world_report.pdf', pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title page
        story.append(Paragraph('State of the World — ' + datetime.now().strftime('%B %Y'), styles['Title']))
        story.append(Spacer(1, 2 * inch))
        story.append(PageBreak())
        
        # Table of contents
        story.append(Paragraph('Table of Contents', styles['Heading1']))
        for topic in TOPICS:
            story.append(Paragraph(topic, styles['Heading2']))
        story.append(PageBreak())
        
        # Sections
        for topic in TOPICS:
            story.append(Paragraph(topic, styles['Heading1']))
            for item in items[topic]:
                story.append(Paragraph(item['title'], styles['Heading2']))
                story.append(Paragraph(item['summary'], styles['BodyText']))
                story.append(Paragraph(f'Source: {item["source_url"]}', styles['Small']))
                story.append(Spacer(1, 0.2 * inch))
            story.append(PageBreak())
        
        doc.build(story)
        logger.info('PDF generated')
    except Exception as e:
        logger.error(f'Error generating PDF: {e}')

# Main execution
def main():
    create_schema()
    nltk.download('punkt')
    
    items = {}
    for topic in TOPICS:
        items[topic] = []
        api_response = get_wiki_api_response(topic)
        if api_response:
            # Extract summary from API response
            summary = api_response['query']['pages'][0]['extract']
            item = {
                'id': str(uuid.uuid4()),
                'topic': topic,
                'title': topic,
                'summary': summary,
                'source_url': f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}',
                'scraped_at': datetime.now(),
                'tokens': tokenize_text(summary)
            }
            items[topic].append(item)
            upsert_item(item)
        else:
            # Fall back to HTML scraping
            html = get_wiki_html(topic)
            if html:
                summary = extract_summary(html)
                item = {
                    'id': str(uuid.uuid4()),
                    'topic': topic,
                    'title': topic,
                    'summary': summary,
                    'source_url': f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}',
                    'scraped_at': datetime.now(),
                    'tokens': tokenize_text(summary)
                }
                items[topic].append(item)
                upsert_item(item)
    
    generate_pdf(items)

if __name__ == '__main__':
    main()