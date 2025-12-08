#!/usr/bin/env python3
"""
State of the World Executive Report Generator

This script scrapes and structures information from Wikipedia across six domains,
stores structured data in PostgreSQL, and generates a professional PDF report.

Designed for Ubuntu 24.04, Python 3.13+, PostgreSQL 17+, and headless execution.
"""

import os
import re
import uuid
import logging
import logging.handlers
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
from urllib.parse import urljoin, quote
import json

import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values
import reportlab.rl_config
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, TOCEntry
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus.tableofcontents import TableOfContents


# ----------------------------
# Constants and Configuration
# ----------------------------

WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1"
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"
REPORT_MONTH = datetime.now(timezone.utc).strftime("%B %Y")
LOG_FILE = "/var/log/state_of_world_report.log"
PDF_OUTPUT_PATH = f"/tmp/State_of_the_World_{datetime.now(timezone.utc).strftime('%Y-%m')}.pdf"

TOPICS = [
    "Emerging technologies",
    "Political shifts",
    "Corporate M&A timelines",
    "Industry trends",
    "Nobel Prizes",
    "Scientific breakthroughs"
]

# Configure ReportLab to be quiet
reportlab.rl_config.warnOnMissingFontGlyphs = 0

# ----------------------------
# Logging Setup
# ----------------------------

def setup_logging():
    """Configure rotating file logger."""
    logger = logging.getLogger("StateOfWorld")
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers if re-run
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5*1024*1024, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logging()


# ----------------------------
# Tokenization Utility
# ----------------------------

def tokenize_text(text: str) -> List[str]:
    """Tokenize input text using simple whitespace + punctuation splitting."""
    if not text:
        return []
    # Remove extra whitespace and split on non-word characters
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


# ----------------------------
# Wikipedia Data Fetchers
# ----------------------------

class WikipediaFetcher:
    """Handles fetching and parsing content from Wikipedia."""

    def __init__(self, session: requests.Session):
        self.session = session
        self.session.headers.update({
            "User-Agent": "StateOfWorldReportBot/1.0 (https://example.com/bot; bot@example.com)"
        })

    def _get_api_page(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch page content via Wikipedia REST API."""
        encoded_title = quote(title.replace(" ", "_"))
        url = f"{WIKIPEDIA_API_BASE}/page/summary/{encoded_title}"
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"API fetch failed for '{title}': {e}")
        return None

    def _scrape_page(self, title: str) -> Optional[Dict[str, Any]]:
        """Fallback HTML scraper using BeautifulSoup."""
        search_url = f"{WIKIPEDIA_BASE_URL}/wiki/{quote(title.replace(' ', '_'))}"
        try:
            resp = self.session.get(search_url, timeout=10)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                return None

            # Extract first paragraph as summary
            first_p = content_div.find("p")
            summary = first_p.get_text(strip=True) if first_p else ""

            # Clean summary
            summary = re.sub(r"\[.*?\]", "", summary)  # Remove citations

            return {
                "title": title,
                "summary": summary,
                "url": search_url
            }
        except Exception as e:
            logger.error(f"Scraping failed for '{title}': {e}")
            return None

    def fetch_topic_content(self, topic: str) -> List[Dict[str, Any]]:
        """Fetch content relevant to a topic."""
        pages_to_try = self._get_candidate_pages(topic)
        items = []

        for page_title in pages_to_try:
            # Try API first
            data = self._get_api_page(page_title)
            if not 
                # Fallback to scraping
                data = self._scrape_page(page_title)

            if data and data.get("summary"):
                clean_summary = re.sub(r"\s+", " ", data["summary"]).strip()
                if len(clean_summary) < 50:  # Skip very short summaries
                    continue

                item = {
                    "id": str(uuid.uuid4()),
                    "topic": topic,
                    "title": data.get("title", page_title),
                    "summary": clean_summary,
                    "source_url": data.get("url") or f"{WIKIPEDIA_BASE_URL}/wiki/{quote(page_title)}",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "tokens": tokenize_text(clean_summary)
                }
                items.append(item)

        return items

    def _get_candidate_pages(self, topic: str) -> List[str]:
        """Map topics to likely Wikipedia page titles."""
        mapping = {
            "Emerging technologies": [
                "Emerging technologies",
                "List of emerging technologies",
                "Technology trends"
            ],
            "Political shifts": [
                "List of coups and coup attempts",
                "Political realignment",
                "Geopolitics"
            ],
            "Corporate M&A timelines": [
                "List of largest mergers and acquisitions",
                "Mergers and acquisitions"
            ],
            "Industry trends": [
                "Industry trend",
                "List of industry trends",
                "Market trend"
            ],
            "Nobel Prizes": [
                "Nobel Prize",
                "List of Nobel laureates",
                "Nobel Prize controversies"
            ],
            "Scientific breakthroughs": [
                "List of scientific breakthroughs",
                "Timeline of scientific discoveries",
                "Scientific revolution"
            ]
        }
        return mapping.get(topic, [topic])


# ----------------------------
# PostgreSQL Database Layer
# ----------------------------

class DatabaseManager:
    """Handles PostgreSQL interactions."""

    def __init__(self):
        self.conn_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "world_report"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD")
        }
        if not self.conn_params["user"] or not self.conn_params["password"]:
            raise ValueError("DB_USER and DB_PASSWORD must be set in environment")

    def create_schema_and_table(self):
        """Create schema and table if they don't exist."""
        create_schema_sql = "CREATE SCHEMA IF NOT EXISTS world_report;"
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS world_report.scraped_items (
            id UUID PRIMARY KEY,
            topic TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            source_url TEXT NOT NULL,
            scraped_at TIMESTAMP WITH TIME ZONE NOT NULL,
            tokens TEXT[]
        );
        """
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(create_schema_sql)
                cur.execute(create_table_sql)
                conn.commit()
        logger.info("Database schema and table ensured.")

    def upsert_items(self, items: List[Dict[str, Any]]) -> int:
        """Upsert items into the database, avoiding duplicates."""
        if not items:
            return 0

        insert_sql = """
        INSERT INTO world_report.scraped_items (
            id, topic, title, summary, source_url, scraped_at, tokens
        ) VALUES %s
        ON CONFLICT (id) DO NOTHING;
        """

        # Convert datetime strings back to datetime objects for psycopg2
        values = []
        for item in items:
            values.append((
                item["id"],
                item["topic"],
                item["title"],
                item["summary"],
                item["source_url"],
                datetime.fromisoformat(item["scraped_at"].replace("Z", "+00:00")),
                item["tokens"]
            ))

        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                execute_values(cur, insert_sql, values, template=None, page_size=100)
                inserted = cur.rowcount
                conn.commit()

        logger.info(f"Upserted {inserted} items into database.")
        return inserted


# ----------------------------
# PDF Report Generator
# ----------------------------

class PDFReportBuilder:
    """Builds a professional PDF report using ReportLab."""

    def __init__(self, items_by_topic: Dict[str, List[Dict[str, Any]]]):
        self.items_by_topic = items_by_topic
        self.styles = getSampleStyleSheet()
        self.custom_styles()

    def custom_styles(self):
        """Define custom paragraph styles."""
        self.styles.add(
            ParagraphStyle(
                name="Title",
                parent=self.styles["Heading1"],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # center
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="Heading2",
                parent=self.styles["Heading2"],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=10
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="Summary",
                parent=self.styles["Normal"],
                fontSize=10,
                leftIndent=20,
                spaceAfter=8
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="Source",
                parent=self.styles["Normal"],
                fontSize=8,
                textColor=colors.grey,
                leftIndent=30,
                spaceAfter=12
            )
        )

    def build(self, output_path: str):
        """Generate the full PDF report."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        story = []

        # Title Page
        title = Paragraph(f"State of the World — {REPORT_MONTH}", self.styles["Title"])
        story.append(title)
        story.append(Spacer(1, 2 * inch))

        # Placeholder for TOC
        toc = TableOfContents()
        toc.levelStyles = [
            self.styles["Heading2"],
            self.styles["Normal"]
        ]
        story.append(toc)
        story.append(PageBreak())

        # Content by Topic
        for topic in TOPICS:
            items = self.items_by_topic.get(topic, [])
            if not items:
                continue

            # Add TOC entry
            story.append(TOCEntry(0, topic))

            story.append(Paragraph(topic, self.styles["Heading2"]))
            story.append(Spacer(1, 12))

            for item in items:
                summary_para = Paragraph(f"• {item['summary']}", self.styles["Summary"])
                source_para = Paragraph(f"Source: {item['source_url']}", self.styles["Source"])
                story.append(summary_para)
                story.append(source_para)

            story.append(Spacer(1, 20))

        # Build PDF
        doc.multiBuild(story)
        logger.info(f"PDF report generated at {output_path}")


# ----------------------------
# Main Execution Logic
# ----------------------------

def fetch_and_store_data(db_manager: DatabaseManager, fetcher: WikipediaFetcher) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch, deduplicate, and store data for all topics."""
    all_items = []
    items_by_topic = {topic: [] for topic in TOPICS}

    for topic in TOPICS:
        logger.info(f"Fetching data for topic: {topic}")
        items = fetcher.fetch_topic_content(topic)
        all_items.extend(items)
        items_by_topic[topic] = items

    # Upsert into DB
    db_manager.upsert_items(all_items)
    return items_by_topic


def main():
    """Main entry point."""
    logger.info("Starting State of the World report generation...")

    try:
        # Initialize components
        session = requests.Session()
        fetcher = WikipediaFetcher(session)
        db_manager = DatabaseManager()
        db_manager.create_schema_and_table()

        # Fetch and store data
        items_by_topic = fetch_and_store_data(db_manager, fetcher)

        # Generate PDF
        pdf_builder = PDFReportBuilder(items_by_topic)
        pdf_builder.build(PDF_OUTPUT_PATH)

        logger.info("Report generation completed successfully.")
    except Exception as e:
        logger.exception(f"Fatal error during report generation: {e}")
        raise


if __name__ == "__main__":
    main()