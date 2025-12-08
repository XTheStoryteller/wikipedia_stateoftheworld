#!/usr/bin/env python3
"""
State of the World Executive Report Generator
Scrapes Wikipedia, stores in PostgreSQL, generates PDF report.
Designed for Ubuntu 24.04, Python 3.13+, PostgreSQL 17+
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
import re
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Third-party imports
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import Json, execute_batch
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import reportlab
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, ListFlowable, ListItem,
    PageTemplate, Frame, NextPageTemplate,
    Image, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# Constants
WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"
USER_AGENT = "StateOfWorldBot/1.0 (https://example.com/bot; contact@example.com)"

# Topic configuration
class Topic(Enum):
    EMERGING_TECH = "Emerging technologies"
    POLITICAL_SHIFTS = "Political shifts"
    CORPORATE_MA = "Corporate M&A timelines"
    INDUSTRY_TRENDS = "Industry trends"
    NOBEL_PRIZES = "Nobel Prizes"
    SCIENTIFIC_BREAKTHROUGHS = "Scientific breakthroughs"

# Data class for scraped items
@dataclass
class ScrapedItem:
    id: str
    topic: str
    title: str
    summary: str
    source_url: str
    scraped_at: datetime
    tokens: List[str]

class WikipediaScraper:
    """Handles Wikipedia API and HTML scraping with fallback logic."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        })
    
    def get_page_content(self, page_title: str) -> Optional[Dict[str, Any]]:
        """Get page content via API first, fall back to HTML scraping."""
        api_url = f"{WIKIPEDIA_API_URL}{page_title}"
        
        try:
            response = self.session.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'extract' in data:
                return {
                    'title': data.get('title', page_title),
                    'summary': data.get('extract', ''),
                    'content_url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'timestamp': data.get('timestamp', datetime.utcnow().isoformat())
                }
        except requests.RequestException as e:
            self.logger.warning(f"API request failed for {page_title}: {e}")
        
        # Fallback to HTML scraping
        return self._scrape_html_page(page_title)
    
    def _scrape_html_page(self, page_title: str) -> Optional[Dict[str, Any]]:
        """Scrape Wikipedia page HTML for content."""
        html_url = f"{WIKIPEDIA_BASE_URL}{page_title.replace(' ', '_')}"
        
        try:
            response = self.session.get(html_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get page title
            title_elem = soup.find('h1', {'id': 'firstHeading'})
            title = title_elem.text.strip() if title_elem else page_title
            
            # Get main content paragraphs
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                # Get first few paragraphs
                paragraphs = content_div.find_all('p', recursive=False)
                summary = ' '.join([p.text.strip() for p in paragraphs[:3] if p.text.strip()])
                
                # Clean up summary
                summary = re.sub(r'\[\d+\]', '', summary)  # Remove citations
                summary = re.sub(r'\s+', ' ', summary).strip()
                
                if len(summary) < 50:  # If too short, try to get more content
                    all_text = content_div.get_text()
                    summary = ' '.join(all_text.split()[:200])
                
                return {
                    'title': title,
                    'summary': summary[:1000],  # Limit length
                    'content_url': html_url,
                    'timestamp': datetime.utcnow().isoformat()
                }
        except Exception as e:
            self.logger.error(f"HTML scraping failed for {page_title}: {e}")
        
        return None
    
    def search_topic(self, topic: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search Wikipedia for pages related to a topic."""
        search_url = "https://en.wikipedia.org/w/api.php"
        
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': f'intitle:{topic}',
            'srlimit': max_results,
            'srprop': 'snippet',
            'utf8': ''
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('query', {}).get('search', []):
                page_content = self.get_page_content(item['title'])
                if page_content:
                    results.append(page_content)
            
            return results
        except Exception as e:
            self.logger.error(f"Search failed for topic {topic}: {e}")
            return []

class DatabaseManager:
    """Manages PostgreSQL database operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.connection = None
        
    def connect(self):
        """Establish database connection using environment variables."""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'world_report_db'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', ''),
                connect_timeout=10
            )
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def create_schema_if_not_exists(self):
        """Create schema and table if they don't exist."""
        create_schema_sql = """
        CREATE SCHEMA IF NOT EXISTS world_report;
        """
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS world_report.scraped_items (
            id UUID PRIMARY KEY,
            topic TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT,
            source_url TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tokens TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(topic, title, source_url)
        );
        
        CREATE INDEX IF NOT EXISTS idx_scraped_items_topic 
        ON world_report.scraped_items(topic);
        
        CREATE INDEX IF NOT EXISTS idx_scraped_items_scraped_at 
        ON world_report.scraped_items(scraped_at);
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_schema_sql)
                cursor.execute(create_table_sql)
            self.logger.info("Schema and table created/verified")
        except Exception as e:
            self.logger.error(f"Schema creation failed: {e}")
            raise
    
    def upsert_items(self, items: List[ScrapedItem]) -> Tuple[int, int]:
        """Insert or update items in the database."""
        if not items:
            return 0, 0
        
        upsert_sql = """
        INSERT INTO world_report.scraped_items 
        (id, topic, title, summary, source_url, scraped_at, tokens)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (topic, title, source_url) 
        DO UPDATE SET
            summary = EXCLUDED.summary,
            scraped_at = EXCLUDED.scraped_at,
            tokens = EXCLUDED.tokens
        RETURNING (xmax = 0) as inserted;
        """
        
        inserted = updated = 0
        
        try:
            with self.connection.cursor() as cursor:
                for item in items:
                    cursor.execute(
                        upsert_sql,
                        (
                            item.id,
                            item.topic,
                            item.title,
                            item.summary,
                            item.source_url,
                            item.scraped_at,
                            item.tokens
                        )
                    )
                    result = cursor.fetchone()
                    if result and result[0]:
                        inserted += 1
                    else:
                        updated += 1
            
            self.connection.commit()
            self.logger.info(f"Database upsert: {inserted} inserted, {updated} updated")
        except Exception as e:
            self.logger.error(f"Database upsert failed: {e}")
            self.connection.rollback()
            raise
        
        return inserted, updated
    
    def get_recent_items(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get items from the last N days, grouped by topic."""
        query_sql = """
        SELECT 
            topic,
            title,
            summary,
            source_url,
            scraped_at,
            tokens
        FROM world_report.scraped_items
        WHERE scraped_at >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY topic, scraped_at DESC;
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query_sql, (days,))
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
            return results
        except Exception as e:
            self.logger.error(f"Failed to fetch recent items: {e}")
            return []
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")

class Tokenizer:
    """Handles text tokenization and normalization."""
    
    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """Tokenize text into words, removing punctuation and stop words."""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove citations and special characters
        text = re.sub(r'\[\d+\]|[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'can', 'may', 'might',
            'must', 'this', 'that', 'these', 'those', 'it', 'its', 'they',
            'them', 'their', 'what', 'which', 'who', 'whom', 'whose'
        }
        
        filtered_tokens = [
            token for token in tokens 
            if token not in stop_words and len(token) > 2
        ]
        
        return filtered_tokens
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by removing extra whitespace and standardizing."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Capitalize first letter of sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.capitalize() for s in sentences if s]
        
        return ' '.join(sentences)

class PDFReportGenerator:
    """Generates the executive PDF report using ReportLab."""
    
    def __init__(self, output_path: str, logger: logging.Logger):
        self.output_path = output_path
        self.logger = logger
        self.styles = getSampleStyleSheet()
        
        # Define custom styles
        self._define_styles()
    
    def _define_styles(self):
        """Define custom styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2C3E50')
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2980B9'),
            borderPadding=5,
            borderColor=colors.HexColor('#3498DB'),
            borderWidth=1,
            borderRadius=3,
            backColor=colors.HexColor('#ECF0F1')
        ))
        
        # Item title style
        self.styles.add(ParagraphStyle(
            name='ItemTitle',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=5,
            textColor=colors.HexColor('#34495E')
        ))
        
        # Summary style
        self.styles.add(ParagraphStyle(
            name='Summary',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#2C3E50')
        ))
        
        # Source URL style
        self.styles.add(ParagraphStyle(
            name='SourceURL',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceAfter=5,
            textColor=colors.HexColor('#7F8C8D'),
            alignment=TA_LEFT
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#95A5A6'),
            alignment=TA_CENTER
        ))
    
    def generate_report(self, data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate PDF report from structured data."""
        try:
            doc = SimpleDocTemplate(
                self.output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72,
                title="State of the World Report"
            )
            
            # Build story
            story = []
            
            # Add title page
            story.extend(self._create_title_page())
            story.append(PageBreak())
            
            # Add table of contents
            story.extend(self._create_table_of_contents(data))
            story.append(PageBreak())
            
            # Add content sections
            for topic_name, items in data.items():
                if items:
                    story.extend(self._create_topic_section(topic_name, items))
                    story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story, onFirstPage=self._add_footer, onLaterPages=self._add_footer)
            
            self.logger.info(f"PDF report generated: {self.output_path}")
            return self.output_path
            
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            raise
    
    def _create_title_page(self) -> List[Any]:
        """Create title page content."""
        elements = []
        
        # Current month/year
        current_date = datetime.now()
        month_year = current_date.strftime("%B %Y")
        
        # Title
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(
            "STATE OF THE WORLD",
            self.styles['MainTitle']
        ))
        elements.append(Paragraph(
            f"Executive Report — {month_year}",
            self.styles['Heading2']
        ))
        
        # Add some spacing
        elements.append(Spacer(1, 1.5*inch))
        
        # Generated timestamp
        generated = f"Generated on: {current_date.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        elements.append(Paragraph(
            generated,
            ParagraphStyle(
                name='Timestamp',
                parent=self.styles['Normal'],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#7F8C8D')
            )
        ))
        
        return elements
    
    def _create_table_of_contents(self, data: Dict[str, List[Dict[str, Any]]]) -> List[Any]:
        """Create table of contents."""
        elements = []
        
        elements.append(Paragraph(
            "Table of Contents",
            self.styles['SectionHeading']
        ))
        elements.append(Spacer(1, 20))
        
        # Create TOC entries
        toc_data = []
        for topic_name, items in data.items():
            if items:
                page_num = 3 + list(data.keys()).index(topic_name)  # Approximate page
                toc_data.append([topic_name, str(page_num)])
        
        if toc_data:
            table = Table(toc_data, colWidths=[4*inch, 1*inch])
            table.setStyle(TableStyle([
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(table)
        
        return elements
    
    def _create_topic_section(self, topic_name: str, items: List[Dict[str, Any]]) -> List[Any]:
        """Create a section for a specific topic."""
        elements = []
        
        # Section heading
        elements.append(Paragraph(
            topic_name,
            self.styles['SectionHeading']
        ))
        elements.append(Spacer(1, 10))
        
        # Add items
        for i, item in enumerate(items[:10], 1):  # Limit to 10 items per topic
            elements.extend(self._create_item_content(item, i))
        
        return elements
    
    def _create_item_content(self, item: Dict[str, Any], index: int) -> List[Any]:
        """Create content for a single item."""
        elements = []
        
        # Item title with number
        title_text = f"{index}. {item['title']}"
        elements.append(Paragraph(
            title_text,
            self.styles['ItemTitle']
        ))
        
        # Summary
        if item.get('summary'):
            # Truncate summary if too long
            summary = item['summary']
            if len(summary) > 500:
                summary = summary[:497] + "..."
            
            elements.append(Paragraph(
                summary,
                self.styles['Summary']
            ))
        
        # Source URL
        if item.get('source_url'):
            elements.append(Paragraph(
                f"Source: {item['source_url']}",
                self.styles['SourceURL']
            ))
        
        # Timestamp
        if item.get('scraped_at'):
            try:
                if isinstance(item['scraped_at'], str):
                    dt = datetime.fromisoformat(item['scraped_at'].replace('Z', '+00:00'))
                else:
                    dt = item['scraped_at']
                
                timestamp = dt.strftime("%Y-%m-%d")
                elements.append(Paragraph(
                    f"Updated: {timestamp}",
                    self.styles['SourceURL']
                ))
            except (ValueError, AttributeError):
                pass
        
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _add_footer(self, canvas, doc):
        """Add footer to each page."""
        canvas.saveState()
        
        # Footer text
        footer_text = f"State of the World Report — Page {doc.page}"
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#95A5A6'))
        
        # Draw footer
        canvas.drawCentredString(
            doc.pagesize[0] / 2,
            30,
            footer_text
        )
        
        # Draw line
        canvas.setStrokeColor(colors.HexColor('#BDC3C7'))
        canvas.setLineWidth(0.5)
        canvas.line(72, 50, doc.pagesize[0] - 72, 50)
        
        canvas.restoreState()

class StateOfWorldReporter:
    """Main orchestrator class."""
    
    def __init__(self):
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.scraper = WikipediaScraper(self.logger)
        self.db = DatabaseManager(self.logger)
        self.tokenizer = Tokenizer()
        
        # Topic-specific search queries
        self.topic_queries = {
            Topic.EMERGING_TECH.value: [
                "artificial intelligence 2024",
                "quantum computing advances",
                "renewable energy technology",
                "biotechnology innovations",
                "space exploration technology"
            ],
            Topic.POLITICAL_SHIFTS.value: [
                "2024 elections",
                "international relations 2024",
                "political developments",
                "diplomatic agreements",
                "government policy changes"
            ],
            Topic.CORPORATE_MA.value: [
                "mergers and acquisitions 2024",
                "corporate takeovers",
                "business consolidation",
                "company acquisitions",
                "corporate restructuring"
            ],
            Topic.INDUSTRY_TRENDS.value: [
                "technology industry trends",
                "economic indicators 2024",
                "market analysis",
                "industry reports",
                "sector growth"
            ],
            Topic.NOBEL_PRIZES.value: [
                "Nobel Prize 2024",
                "Nobel laureates",
                "Nobel Prize winners",
                "Nobel Prize categories",
                "Nobel foundation"
            ],
            Topic.SCIENTIFIC_BREAKTHROUGHS.value: [
                "scientific discoveries 2024",
                "medical breakthroughs",
                "physics discoveries",
                "climate change research",
                "astronomy findings"
            ]
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configure rotating file logging."""
        logger = logging.getLogger('StateOfWorldReporter')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        log_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'state_of_world.log'
        )
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def scrape_topic(self, topic: str) -> List[ScrapedItem]:
        """Scrape content for a specific topic."""
        self.logger.info(f"Scraping topic: {topic}")
        
        scraped_items = []
        queries = self.topic_queries.get(topic, [topic])
        
        for query in queries:
            try:
                # Search Wikipedia for related pages
                search_results = self.scraper.search_topic(query, max_results=3)
                
                for result in search_results:
                    # Tokenize the summary
                    tokens = self.tokenizer.tokenize_text(result.get('summary', ''))
                    
                    # Create ScrapedItem
                    item = ScrapedItem(
                        id=str(uuid.uuid4()),
                        topic=topic,
                        title=result.get('title', query),
                        summary=self.tokenizer.normalize_text(result.get('summary', '')),
                        source_url=result.get('content_url', ''),
                        scraped_at=datetime.utcnow(),
                        tokens=tokens
                    )
                    
                    # Only include if we have meaningful content
                    if item.summary and len(item.summary) > 50:
                        scraped_items.append(item)
                    
                    # Limit to avoid too many requests
                    if len(scraped_items) >= 5:
                        break
                        
            except Exception as e:
                self.logger.error(f"Error scraping query '{query}' for topic '{topic}': {e}")
                continue
        
        self.logger.info(f"Scraped {len(scraped_items)} items for topic: {topic}")
        return scraped_items
    
    def generate_monthly_report(self) -> str:
        """Main method to generate monthly report."""
        self.logger.info("Starting monthly State of the World report generation")
        
        try:
            # Connect to database
            self.db.connect()
            self.db.create_schema_if_not_exists()
            
            # Scrape all topics
            all_items = []
            for topic in [t.value for t in Topic]:
                items = self.scrape_topic(topic)
                all_items.extend(items)
            
            # Store in database
            if all_items:
                inserted, updated = self.db.upsert_items(all_items)
                self.logger.info(f"Total database operations: {inserted} inserted, {updated} updated")
            else:
                self.logger.warning("No items scraped")
            
            # Get recent items (last 30 days)
            recent_items = self.db.get_recent_items(days=30)
            
            # Group by topic
            grouped_data = {}
            for topic in [t.value for t in Topic]:
                topic_items = [item for item in recent_items if item['topic'] == topic]
                if topic_items:
                    grouped_data[topic] = topic_items
            
            # Generate PDF report
            if grouped_data:
                # Create output directory if needed
                output_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'reports'
                )
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"state_of_world_{timestamp}.pdf"
                pdf_path = os.path.join(output_dir, pdf_filename)
                
                pdf_generator = PDFReportGenerator(pdf_path, self.logger)
                pdf_path = pdf_generator.generate_report(grouped_data)
                
                self.logger.info(f"Report generation complete: {pdf_path}")
                return pdf_path
            else:
                self.logger.warning("No recent data to generate report")
                return ""
                
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
        finally:
            self.db.close()
    
    def run(self):
        """Main execution method."""
        try:
            report_path = self.generate_monthly_report()
            if report_path:
                self.logger.info(f"Report successfully generated: {report_path}")
                print(f"Report generated: {report_path}")
            else:
                self.logger.warning("No report generated")
                print("No report generated - check logs for details")
                
        except KeyboardInterrupt:
            self.logger.info("Process interrupted by user")
            print("\nProcess interrupted")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            print(f"Error: {e}")
            sys.exit(1)

def main():
    """Entry point for the script."""
    # Check Python version
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required")
        sys.exit(1)
    
    # Check required environment variables
    required_env_vars = ['DB_NAME', 'DB_USER']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the script")
        sys.exit(1)
    
    # Run the reporter
    reporter = StateOfWorldReporter()
    reporter.run()

if __name__ == "__main__":
    main()