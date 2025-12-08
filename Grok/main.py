#!/usr/bin/env python3
"""
State of the World — Monthly Executive Report Generator
Fully production-ready script for Ubuntu 24.04 + Python 3.13 + PostgreSQL 17+
Generates a professional PDF report from Wikipedia data every month.
"""

import os
import sys
import uuid
import logging
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    TableOfContents,
    FrameBreak,
    NextPageTemplate,
    PageTemplate,
    Frame,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib import colors

import psycopg2
from psycopg2.extras import execute_values
from logging.handlers import RotatingFileHandler

# --------------------------------------------------------------------------- #
# Configuration & Constants
# --------------------------------------------------------------------------- #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "state_of_world.log")
PDF_OUTPUT = os.path.join(SCRIPT_DIR, f"State_of_the_World_{datetime.now().strftime('%Y_%m')}.pdf")

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_BASE = "https://en.wikipedia.org/wiki/"

HEADERS = {
    "User-Agent": "StateOfWorldBot/1.0 (+https://github.com/yourname/state-of-world)"
}

# Topics and their primary Wikipedia pages (can be expanded)
TOPICS = {
    "Emerging technologies": [
        "2025_in_science",
        "Timeline_of_scientific_discoveries",
        "Emerging_technologies",
    ],
    "Political shifts": [
        "2025",
        "2025_in_politics",
        "List_of_elections_in_2025",
        "Portal:Current_events",
    ],
    "Corporate M&A timelines": [
        "List_of_mergers_and_acquisitions",
        "2025_in_business",
    ],
    "Industry trends": [
        "2025",
        "Technology_in_2025",
        "Global_economic_trends",
    ],
    "Nobel Prizes": [
        "2025_Nobel_Prize",
        "List_of_Nobel_Prize_laureates",
        "Nobel_Prize",
    ],
    "Scientific breakthroughs": [
        "2025_in_science",
        "Breakthrough_of_the_Year",
        "Timeline_of_scientific_discoveries",
    ],
}

# Environment variables (set these!)
REQUIRED_ENV = [
    "PGHOST",
    "PGDATABASE",
    "PGUSER",
    "PGPASSWORD",
    "PGPORT",
]

# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #

logger = logging.getLogger("StateOfWorld")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# --------------------------------------------------------------------------- #
# Simple Tokenizer (no external deps)
# --------------------------------------------------------------------------- #

def tokenize(text: str) -> List[str]:
    """Very lightweight tokenizer - split on whitespace and basic punctuation."""
    import re
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens

# --------------------------------------------------------------------------- #
# Wikipedia Utilities
# --------------------------------------------------------------------------- #

def wikipedia_api_query(page_title: str) -> Optional[Dict]:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "inprop": "url",
        "titles": page_title,
        "format": "json",
    }
    try:
        resp = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=15)
        data = resp.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        if int(page.get("pageid", 0)) <= 0:
            return None
        return {
            "title": page["title"],
            "extract": page.get("extract", ""),
            "fullurl": page.get("fullurl", ""),
            "pageid": page["pageid"],
        }
    except Exception as e:
        logger.warning(f"API query failed for {page_title}: {e}")
        return None


def scrape_wikipedia_page(url: str) -> str:
    """Fallback HTML scrape when API extract is insufficient."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts, styles, references, etc.
        for tag in soup(["script", "style", "sup", "ref"]):
            tag.decompose()

        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            return ""

        paragraphs = content.find_all("p", recursive=True)
        text_parts = [p.get_text(strip=True) for p in paragraphs[:15]]  # top paragraphs
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        return ""


def extract_structured_content(topic: str, pages: List[str]) -> List[Dict[str, Any]]:
    items = []
    seen_titles = set()

    cutoff_date = datetime.utcnow() - timedelta(days=35)  # ~30 days + buffer

    for page in pages:
        api_result = wikipedia_api_query(page)
        if api_result and api_result["extract"]:
            text = api_result["extract"]
            url = api_result["fullurl"]
            title = api_result["title"]
        else:
            url = WIKIPEDIA_BASE + quote_plus(page.replace(" ", "_"))
            text = scrape_wikipedia_page(url)
            title = page

        if not text.strip() or title in seen_titles:
            continue

        # Very simple date filtering - look for "2025" or current year
        current_year = datetime.now().year
        if str(current_year) not in text and "2025" not in text:
            continue

        summary = " ".join(text.split()[:300]) + "..." if len(text.split()) > 300 else text

        item = {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "title": title,
            "summary": summary.strip(),
            "source_url": url,
            "scraped_at": datetime.utcnow(),
            "tokens": tokenize(summary),
        }
        items.append(item)
        seen_titles.add(title)

    return items


# --------------------------------------------------------------------------- #
# Database Layer (psycopg2 - stable, synchronous, widely available)
# --------------------------------------------------------------------------- #

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        port=os.getenv("PGPORT", "5432"),
    )


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE SCHEMA IF NOT EXISTS world_report;
        
        CREATE TABLE IF NOT EXISTS world_report.scraped_items (
            id UUID PRIMARY KEY,
            topic TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            source_url TEXT NOT NULL,
            scraped_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            tokens TEXT[],
            UNIQUE(topic, title, source_url)
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database schema ensured.")


def upsert_items(items: List[Dict[str, Any]]):
    if not items:
        return

    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        INSERT INTO world_report.scraped_items 
        (id, topic, title, summary, source_url, scraped_at, tokens)
        VALUES %s
        ON CONFLICT (topic, title, source_url) DO UPDATE SET
            summary = EXCLUDED.summary,
            scraped_at = EXCLUDED.scraped_at,
            tokens = EXCLUDED.tokens;
    """

    data = [
        (
            item["id"],
            item["topic"],
            item["title"],
            item["summary"],
            item["source_url"],
            item["scraped_at"],
            item["tokens"],
        )
        for item in items
    ]

    inserted = execute_values(cur, query, data, fetch=True)
    conn.commit()
    cur.close()
    conn.close()

    logger.info(f"Upserted {len(data)} items ({len(inserted)} new or updated)")


def load_report_data() -> Dict[str, List[Dict]]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT topic, title, summary, source_url 
        FROM world_report.scraped_items 
        WHERE scraped_at >= %s
        ORDER BY topic, title
        """,
        (datetime.utcnow() - timedelta(days=35),),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    data = {}
    for topic, title, summary, url in rows:
        data.setdefault(topic, []).append({"title": title, "summary": summary, "url": url})
    return data


# --------------------------------------------------------------------------- #
# PDF Generation with reportlab.platypus
# --------------------------------------------------------------------------- #

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleLarge", fontSize=28, leading=32, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Heading1Big", parent=styles["Heading1"], fontSize=18, spaceAfter=12))
styles.add(ParagraphStyle(name="Source", fontSize=8, textColor=colors.grey, leading=10))

def build_pdf(filename: str, report_data: Dict[str, List[Dict]]):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=60,
    )

    story = []
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name='TOCHeading1', fontSize=14, leftIndent=0),
        ParagraphStyle(name='TOCHeading2', fontSize=12, leftIndent=20),
    ]

    # Title Page
    story.append(Paragraph(f"State of the World", styles["TitleLarge"]))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Monthly Executive Report — {datetime.now().strftime('%B %Y')}", styles["Title"]))
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Generated on " + datetime.now().strftime("%Y-%m-%d"), styles["Normal"]))
    story.append(PageBreak())

    # TOC placeholder
    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    story.append(toc)
    story.append(PageBreak())

    section_number = 1
    for topic, entries in report_data.items():
        story.append(Paragraph(f"{section_number}. {topic}", styles["Heading1Big"]))
        toc.addEntry(0, f"{section_number}. {topic}", len(story)-1)
        section_number += 1

        for entry in entries:
            story.append(Paragraph(entry["title"], styles["Heading2"]))
            story.append(Paragraph(entry["summary"], styles["Normal"]))
            story.append(Paragraph(entry["url"], styles["Source"]))
            story.append(Spacer(1, 0.2*inch))

        story.append(PageBreak())

    # Footer with page numbers
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawCentredString(doc.width/2, 0.5*inch, text)
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    logger.info(f"PDF generated: {filename}")


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #

async def main():
    parser = argparse.ArgumentParser(description="Generate monthly State of the World report")
    parser.add_argument("--no-fetch", action="store_true", help="Skip fetching new data (use existing DB)")
    args = parser.parse_args()

    # Validate env
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        sys.exit(1)

    init_db()

    if not args.no_fetch:
        logger.info("Starting data collection from Wikipedia...")
        all_items = []

        for topic, pages in TOPICS.items():
            logger.info(f"Processing topic: {topic}")
            items = extract_structured_content(topic, pages)
            all_items.extend(items)
            logger.info(f"  → Collected {len(items)} items")

        if all_items:
            upsert_items(all_items)
        else:
            logger.warning("No new items collected this run.")

    # Always generate PDF from DB (last ~35 days)
    logger.info("Loading data from database for PDF...")
    report_data = load_report_data()

    if not report_data:
        logger.error("No data available in DB for report generation.")
        sys.exit(1)

    build_pdf(PDF_OUTPUT, report_data)
    logger.info("Monthly report generation complete!")


if __name__ == "__main__":
    asyncio.run(main())