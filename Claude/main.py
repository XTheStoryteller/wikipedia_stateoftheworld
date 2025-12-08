#!/usr/bin/env python3
"""
State of the World - Monthly Executive Report Generator
Scrapes Wikipedia for 6 key topics, stores in PostgreSQL, generates PDF report

Requirements:
- Ubuntu 24.04
- Python 3.13+
- PostgreSQL 17+
- Dependencies: requests, beautifulsoup4, psycopg2-binary, reportlab

Author: Senior Python Engineer
License: MIT
"""

import os
import sys
import logging
import uuid
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import connection

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, PageTemplate, Frame, NextPageTemplate
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.doctemplate import BaseDocTemplate


# ============================================================================
# CONSTANTS
# ============================================================================

LOG_FILE = "world_report.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1"
WIKIPEDIA_WEB_BASE = "https://en.wikipedia.org"
USER_AGENT = "StateOfWorldReport/1.0 (Educational; Python/3.13)"

# Topic definitions with Wikipedia search queries and keywords
TOPICS = {
    "Emerging Technologies": {
        "queries": [
            "Emerging technologies",
            "Artificial intelligence",
            "Quantum computing",
            "Biotechnology"
        ],
        "keywords": ["technology", "innovation", "AI", "quantum", "biotech"]
    },
    "Political Shifts": {
        "queries": [
            "2024 in politics",
            "2025 in politics",
            "Political movements",
            "International relations"
        ],
        "keywords": ["election", "government", "policy", "diplomatic", "political"]
    },
    "Corporate M&A": {
        "queries": [
            "List of mergers and acquisitions",
            "2024 mergers and acquisitions",
            "Corporate acquisitions"
        ],
        "keywords": ["merger", "acquisition", "buyout", "takeover", "deal"]
    },
    "Industry Trends": {
        "queries": [
            "Economic trends",
            "Industry analysis",
            "Market trends",
            "Global economy"
        ],
        "keywords": ["industry", "market", "economic", "trend", "growth"]
    },
    "Nobel Prizes": {
        "queries": [
            "Nobel Prize",
            "2024 Nobel Prize",
            "List of Nobel laureates"
        ],
        "keywords": ["nobel", "laureate", "prize", "awarded"]
    },
    "Scientific Breakthroughs": {
        "queries": [
            "Scientific discoveries",
            "Breakthroughs in science",
            "Recent scientific developments",
            "2024 in science"
        ],
        "keywords": ["discovery", "breakthrough", "research", "scientific", "study"]
    }
}

DB_SCHEMA = "world_report"
DB_TABLE = "scraped_items"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure rotating file and console logging"""
    logger = logging.getLogger("WorldReport")
    logger.setLevel(logging.INFO)
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ScrapedItem:
    """Represents a scraped content item"""
    id: str
    topic: str
    title: str
    summary: str
    source_url: str
    scraped_at: datetime
    tokens: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database insertion"""
        return {
            'id': self.id,
            'topic': self.topic,
            'title': self.title,
            'summary': self.summary,
            'source_url': self.source_url,
            'scraped_at': self.scraped_at,
            'tokens': self.tokens
        }


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization: lowercase, split on whitespace and punctuation
    Returns list of meaningful tokens (words)
    """
    if not text:
        return []
    
    # Lowercase and remove special characters except spaces
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split on whitespace and filter out short tokens
    tokens = [t.strip() for t in text.split() if len(t.strip()) > 2]
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
        'her', 'was', 'one', 'our', 'out', 'day', 'has', 'him', 'his',
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who',
        'with', 'from', 'have', 'this', 'that', 'will', 'what', 'when',
        'been', 'into', 'than', 'them', 'then', 'were', 'said'
    }
    
    tokens = [t for t in tokens if t not in stop_words]
    
    return tokens[:100]  # Limit to 100 tokens per item


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove citation markers
    text = re.sub(r'\[\d+\]', '', text)
    # Trim
    text = text.strip()
    
    return text


def truncate_summary(text: str, max_length: int = 500) -> str:
    """Truncate text to maximum length, preserving whole sentences"""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > max_length * 0.7:  # If we have a sentence end in the last 30%
        return truncated[:last_period + 1]
    else:
        return truncated + "..."


# ============================================================================
# WIKIPEDIA SCRAPING
# ============================================================================

class WikipediaScraper:
    """Handles Wikipedia API and web scraping"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
    
    def search_wikipedia(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Wikipedia using REST API"""
        try:
            url = f"{WIKIPEDIA_API_BASE}/page/search/{quote(query)}"
            params = {'limit': limit}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('pages', [])
            
        except Exception as e:
            logger.error(f"Wikipedia search failed for '{query}': {e}")
            return []
    
    def get_page_summary(self, title: str) -> Optional[Dict]:
        """Get page summary from Wikipedia REST API"""
        try:
            url = f"{WIKIPEDIA_API_BASE}/page/summary/{quote(title)}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get summary for '{title}': {e}")
            return None
    
    def scrape_page_html(self, title: str) -> Optional[str]:
        """Scrape full HTML content from Wikipedia page"""
        try:
            url = f"{WIKIPEDIA_WEB_BASE}/wiki/{quote(title.replace(' ', '_'))}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
            
            # Extract paragraphs from the main content
            paragraphs = content_div.find_all('p', limit=10)
            text_parts = []
            
            for p in paragraphs:
                text = p.get_text().strip()
                if text and len(text) > 50:  # Meaningful paragraphs only
                    text_parts.append(text)
            
            return ' '.join(text_parts)
            
        except Exception as e:
            logger.error(f"Failed to scrape HTML for '{title}': {e}")
            return None
    
    def extract_lists_from_html(self, title: str) -> List[str]:
        """Extract list items from Wikipedia page (for timelines, etc.)"""
        try:
            url = f"{WIKIPEDIA_WEB_BASE}/wiki/{quote(title.replace(' ', '_'))}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find('div', {'id': 'mw-content-text'})
            
            if not content_div:
                return []
            
            items = []
            # Find all list items
            for li in content_div.find_all('li', limit=50):
                text = li.get_text().strip()
                if text and len(text) > 20 and len(text) < 500:
                    items.append(clean_text(text))
            
            return items[:20]  # Return top 20 items
            
        except Exception as e:
            logger.error(f"Failed to extract lists from '{title}': {e}")
            return []


# ============================================================================
# CONTENT EXTRACTION
# ============================================================================

class ContentExtractor:
    """Extracts and structures content for each topic"""
    
    def __init__(self):
        self.scraper = WikipediaScraper()
    
    def extract_for_topic(self, topic: str, topic_config: Dict) -> List[ScrapedItem]:
        """Extract content items for a specific topic"""
        logger.info(f"Extracting content for topic: {topic}")
        
        items = []
        queries = topic_config['queries']
        
        for query in queries:
            # Search Wikipedia
            search_results = self.scraper.search_wikipedia(query, limit=3)
            
            for result in search_results:
                title = result.get('title', '')
                if not title:
                    continue
                
                # Get summary via API
                summary_data = self.scraper.get_page_summary(title)
                
                if summary_data:
                    summary = summary_data.get('extract', '')
                    page_url = summary_data.get('content_urls', {}).get('desktop', {}).get('page', '')
                else:
                    # Fallback to HTML scraping
                    summary = self.scraper.scrape_page_html(title)
                    page_url = f"{WIKIPEDIA_WEB_BASE}/wiki/{quote(title.replace(' ', '_'))}"
                
                if not summary:
                    continue
                
                # Clean and truncate
                summary = clean_text(summary)
                summary = truncate_summary(summary, max_length=600)
                
                # Tokenize
                tokens = tokenize_text(summary)
                
                # Create item
                item = ScrapedItem(
                    id=str(uuid.uuid4()),
                    topic=topic,
                    title=title,
                    summary=summary,
                    source_url=page_url,
                    scraped_at=datetime.now(),
                    tokens=tokens
                )
                
                items.append(item)
                logger.info(f"  Extracted: {title}")
            
            # For certain topics, also extract list items
            if topic in ["Corporate M&A", "Nobel Prizes"]:
                for result in search_results[:1]:  # Just the top result
                    title = result.get('title', '')
                    list_items = self.scraper.extract_lists_from_html(title)
                    
                    for idx, list_text in enumerate(list_items[:5]):
                        item = ScrapedItem(
                            id=str(uuid.uuid4()),
                            topic=topic,
                            title=f"{title} - Item {idx + 1}",
                            summary=list_text,
                            source_url=f"{WIKIPEDIA_WEB_BASE}/wiki/{quote(title.replace(' ', '_'))}",
                            scraped_at=datetime.now(),
                            tokens=tokenize_text(list_text)
                        )
                        items.append(item)
        
        logger.info(f"Extracted {len(items)} items for {topic}")
        return items


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

class DatabaseManager:
    """Manages PostgreSQL database operations"""
    
    def __init__(self):
        self.conn_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'world_report'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
    
    def get_connection(self) -> connection:
        """Create database connection"""
        return psycopg2.connect(**self.conn_params)
    
    def initialize_schema(self):
        """Create schema and tables if they don't exist"""
        logger.info("Initializing database schema...")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Create schema
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA}")
                
                # Create table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.{DB_TABLE} (
                        id UUID PRIMARY KEY,
                        topic TEXT NOT NULL,
                        title TEXT NOT NULL,
                        summary TEXT,
                        source_url TEXT,
                        scraped_at TIMESTAMP NOT NULL,
                        tokens TEXT[],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(topic, title)
                    )
                """)
                
                # Create indices
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_scraped_items_topic 
                    ON {DB_SCHEMA}.{DB_TABLE}(topic)
                """)
                
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_scraped_items_scraped_at 
                    ON {DB_SCHEMA}.{DB_TABLE}(scraped_at)
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize schema: {e}")
            raise
        finally:
            conn.close()
    
    def upsert_items(self, items: List[ScrapedItem]) -> Tuple[int, int]:
        """
        Insert or update items in database
        Returns: (inserted_count, updated_count)
        """
        if not items:
            return 0, 0
        
        logger.info(f"Upserting {len(items)} items to database...")
        
        conn = self.get_connection()
        inserted = 0
        updated = 0
        
        try:
            with conn.cursor() as cur:
                for item in items:
                    item_dict = item.to_dict()
                    
                    # Check if exists
                    cur.execute(f"""
                        SELECT id FROM {DB_SCHEMA}.{DB_TABLE}
                        WHERE topic = %s AND title = %s
                    """, (item.topic, item.title))
                    
                    exists = cur.fetchone()
                    
                    if exists:
                        # Update existing
                        cur.execute(f"""
                            UPDATE {DB_SCHEMA}.{DB_TABLE}
                            SET summary = %s,
                                source_url = %s,
                                scraped_at = %s,
                                tokens = %s
                            WHERE topic = %s AND title = %s
                        """, (
                            item.summary,
                            item.source_url,
                            item.scraped_at,
                            item.tokens,
                            item.topic,
                            item.title
                        ))
                        updated += 1
                    else:
                        # Insert new
                        cur.execute(f"""
                            INSERT INTO {DB_SCHEMA}.{DB_TABLE}
                            (id, topic, title, summary, source_url, scraped_at, tokens)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            item.id,
                            item.topic,
                            item.title,
                            item.summary,
                            item.source_url,
                            item.scraped_at,
                            item.tokens
                        ))
                        inserted += 1
                
                conn.commit()
                logger.info(f"Database upsert complete: {inserted} inserted, {updated} updated")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Database upsert failed: {e}")
            raise
        finally:
            conn.close()
        
        return inserted, updated
    
    def get_items_for_period(self, days: int = 30) -> Dict[str, List[Dict]]:
        """Retrieve items from the last N days, grouped by topic"""
        logger.info(f"Retrieving items from last {days} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = self.get_connection()
        items_by_topic = {}
        
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT topic, title, summary, source_url, scraped_at
                    FROM {DB_SCHEMA}.{DB_TABLE}
                    WHERE scraped_at >= %s
                    ORDER BY topic, scraped_at DESC
                """, (cutoff_date,))
                
                rows = cur.fetchall()
                
                for row in rows:
                    topic, title, summary, source_url, scraped_at = row
                    
                    if topic not in items_by_topic:
                        items_by_topic[topic] = []
                    
                    items_by_topic[topic].append({
                        'title': title,
                        'summary': summary,
                        'source_url': source_url,
                        'scraped_at': scraped_at
                    })
                
                logger.info(f"Retrieved {len(rows)} items across {len(items_by_topic)} topics")
                
        except Exception as e:
            logger.error(f"Failed to retrieve items: {e}")
            raise
        finally:
            conn.close()
        
        return items_by_topic


# ============================================================================
# PDF GENERATION
# ============================================================================

class PDFReportGenerator:
    """Generates professional PDF reports using ReportLab Platypus"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.story = []
        self.toc = TableOfContents()
    
    def _setup_custom_styles(self):
        """Define custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Topic heading
        self.styles.add(ParagraphStyle(
            name='TopicHeading',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Item title
        self.styles.add(ParagraphStyle(
            name='ItemTitle',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=6,
            fontName='Helvetica-Bold',
            bulletIndent=20,
            leftIndent=20
        ))
        
        # Summary text
        self.styles.add(ParagraphStyle(
            name='Summary',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#555555'),
            spaceAfter=8,
            leftIndent=40,
            rightIndent=20,
            alignment=TA_LEFT
        ))
        
        # Source URL
        self.styles.add(ParagraphStyle(
            name='SourceURL',
            parent=self.styles['BodyText'],
            fontSize=8,
            textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=16,
            leftIndent=40,
            fontName='Helvetica-Oblique'
        ))
    
    def add_title_page(self, report_month: str):
        """Add title page"""
        self.story.append(Spacer(1, 2*inch))
        
        title = Paragraph(
            f"State of the World",
            self.styles['CustomTitle']
        )
        self.story.append(title)
        self.story.append(Spacer(1, 0.3*inch))
        
        subtitle = Paragraph(
            f"Executive Report — {report_month}",
            self.styles['Heading2']
        )
        subtitle.style.alignment = TA_CENTER
        self.story.append(subtitle)
        
        self.story.append(Spacer(1, 0.5*inch))
        
        timestamp = Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y')}",
            self.styles['Normal']
        )
        timestamp.style.alignment = TA_CENTER
        timestamp.style.textColor = colors.HexColor('#7f8c8d')
        self.story.append(timestamp)
        
        self.story.append(PageBreak())
    
    def add_table_of_contents(self):
        """Add table of contents"""
        toc_title = Paragraph("Table of Contents", self.styles['Heading1'])
        self.story.append(toc_title)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Configure TOC
        self.toc.levelStyles = [
            ParagraphStyle(
                name='TOCLevel1',
                fontSize=12,
                leftIndent=20,
                spaceAfter=8,
                textColor=colors.HexColor('#2c3e50')
            )
        ]
        
        self.story.append(self.toc)
        self.story.append(PageBreak())
    
    def add_topic_section(self, topic: str, items: List[Dict]):
        """Add a topic section with items"""
        # Topic heading (automatically added to TOC)
        heading = Paragraph(topic, self.styles['TopicHeading'])
        self.story.append(heading)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Add each item
        for idx, item in enumerate(items, 1):
            # Item title
            title_text = f"• {item['title']}"
            title = Paragraph(title_text, self.styles['ItemTitle'])
            self.story.append(title)
            
            # Summary
            summary = Paragraph(item['summary'], self.styles['Summary'])
            self.story.append(summary)
            
            # Source URL
            source = Paragraph(
                f"Source: {item['source_url']}",
                self.styles['SourceURL']
            )
            self.story.append(source)
            
            # Separator between items
            if idx < len(items):
                self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(PageBreak())
    
    def generate(self, items_by_topic: Dict[str, List[Dict]]):
        """Generate the complete PDF report"""
        logger.info(f"Generating PDF report: {self.filename}")
        
        # Calculate report month
        report_date = datetime.now() - timedelta(days=15)
        report_month = report_date.strftime('%B %Y')
        
        # Build story
        self.add_title_page(report_month)
        self.add_table_of_contents()
        
        # Add topic sections
        for topic in TOPICS.keys():
            items = items_by_topic.get(topic, [])
            if items:
                self.add_topic_section(topic, items)
            else:
                # Add placeholder if no items
                heading = Paragraph(topic, self.styles['TopicHeading'])
                self.story.append(heading)
                self.story.append(Spacer(1, 0.2*inch))
                
                no_data = Paragraph(
                    "No data available for this period.",
                    self.styles['Summary']
                )
                self.story.append(no_data)
                self.story.append(PageBreak())
        
        # Build PDF
        doc = SimpleDocTemplate(
            self.filename,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Add page numbers
        def add_page_number(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            page_num = canvas.getPageNumber()
            text = f"Page {page_num}"
            canvas.drawRightString(letter[0] - 72, 40, text)
            canvas.restoreState()
        
        doc.build(self.story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        
        logger.info(f"PDF report generated successfully: {self.filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("State of the World - Monthly Report Generation Started")
    logger.info("="*80)
    
    try:
        # Initialize database
        db_manager = DatabaseManager()
        db_manager.initialize_schema()
        
        # Extract content for all topics
        extractor = ContentExtractor()
        all_items = []
        
        for topic, config in TOPICS.items():
            items = extractor.extract_for_topic(topic, config)
            all_items.extend(items)
        
        logger.info(f"Total items extracted: {len(all_items)}")
        
        # Store in database
        if all_items:
            inserted, updated = db_manager.upsert_items(all_items)
            logger.info(f"Database operations: {inserted} inserted, {updated} updated")
        
        # Retrieve items for report (last 30 days)
        items_by_topic = db_manager.get_items_for_period(days=30)
        
        # Generate PDF report
        report_filename = f"state_of_world_{datetime.now().strftime('%Y%m')}.pdf"
        pdf_generator = PDFReportGenerator(report_filename)
        pdf_generator.generate(items_by_topic)
        
        logger.info("="*80)
        logger.info(f"Report generation completed successfully!")
        logger.info(f"PDF Report: {report_filename}")
        logger.info("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
