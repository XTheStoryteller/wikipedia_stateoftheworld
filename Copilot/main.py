#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly "State of the World" executive report generator.

Features:
- Queries Wikipedia REST API first; falls back to HTML scraping via BeautifulSoup if needed.
- Extracts structured sections, key paragraphs, lists, and timelines for six topics.
- Cleans, normalizes, and tokenizes content (built-in Python only).
- Stores items in PostgreSQL (schema: world_report, table: scraped_items) with UPSERT.
- Runs idempotently with deterministic UUIDs; respects a 30-day window via Wikipedia timestamps or Last-Modified.
- Generates a professional PDF via reportlab.platypus, including TOC and page numbers.
- Safe for cron execution; logs via RotatingFileHandler.

Environment:
- Ubuntu 24.04
- Python 3.13+
- PostgreSQL 17+ (psycopg2)
- reportlab, requests, bs4

Author: Senior Python Engineer
"""

from __future__ import annotations

import os
import sys
import re
import uuid
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote

import logging
from logging.handlers import RotatingFileHandler

import requests
from bs4 import BeautifulSoup

import psycopg2
from psycopg2.extras import execute_values


# =========================
# Constants and configuration
# =========================

LOG_FILE = os.environ.get("WORLD_REPORT_LOG_FILE", os.path.abspath("world_report.log"))
LOG_MAX_BYTES = int(os.environ.get("WORLD_REPORT_LOG_MAX_BYTES", str(2 * 1024 * 1024)))  # 2MB
LOG_BACKUP_COUNT = int(os.environ.get("WORLD_REPORT_LOG_BACKUP_COUNT", "3"))
DEFAULT_TIMEOUT = (10, 30)  # (connect, read) seconds

# Wikipedia REST API endpoints
WIKI_API_BASE = "https://en.wikipedia.org/api/rest_v1"
WIKI_WEB_BASE = "https://en.wikipedia.org/wiki"

# Monthly window: today minus 30 days
NOW_UTC = datetime.now(timezone.utc)
WINDOW_START = NOW_UTC - timedelta(days=30)

# Topics and queries (primary & fallback query terms)
TOPIC_QUERIES: Dict[str, List[str]] = {
    "Emerging technologies": [
        "Emerging technologies",
        "Artificial intelligence",
        "Quantum computing",
        "Biotechnology",
        "Nanotechnology",
    ],
    "Political shifts": [
        "Politics",
        "Geopolitics",
        "Government",
        "Elections",
        "International relations",
        "Current events portal",
    ],
    "Corporate M&A timelines": [
        "Mergers and acquisitions",
        "Corporate mergers",
        "Acquisition",
        "List of mergers and acquisitions",
        "Private equity",
    ],
    "Industry trends": [
        "Industry",
        "Economic trends",
        "Business trends",
        "Macroeconomics",
        "Digital transformation",
    ],
    "Nobel Prizes": [
        "Nobel Prize",
        "List of Nobel laureates",
        "Nobel Prize in Physics",
        "Nobel Prize in Chemistry",
        "Nobel Prize in Medicine",
        "Nobel Prize in Literature",
        "Nobel Peace Prize",
        "Nobel Prize in Economic Sciences",
    ],
    "Scientific breakthroughs": [
        "Science",
        "Breakthrough",
        "List of scientific discoveries",
        "2025 in science",
        "Technology breakthroughs",
    ],
}

# PostgreSQL environment variables
DB_HOST = os.environ.get("WORLD_DB_HOST", "localhost")
DB_PORT = int(os.environ.get("WORLD_DB_PORT", "5432"))
DB_NAME = os.environ.get("WORLD_DB_NAME", "world")
DB_USER = os.environ.get("WORLD_DB_USER", "world_user")
DB_PASSWORD = os.environ.get("WORLD_DB_PASSWORD", "")
DB_SSLMODE = os.environ.get("WORLD_DB_SSLMODE", "prefer")  # prefer, require, disable

DB_SCHEMA = "world_report"
DB_TABLE = "scraped_items"

# PDF output
OUTPUT_DIR = os.environ.get("WORLD_REPORT_OUTPUT_DIR", os.path.abspath("./output"))
PDF_FILENAME = f"State_of_the_World_{NOW_UTC.strftime('%Y_%m')}.pdf"
PDF_PATH = os.path.join(OUTPUT_DIR, PDF_FILENAME)


# =========================
# Logging setup
# =========================

logger = logging.getLogger("world_report")
logger.setLevel(logging.INFO)
_log_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_log_handler.setFormatter(_log_formatter)
logger.addHandler(_log_handler)


# =========================
# Utility functions
# =========================

def safe_request_json(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    """Perform a GET request expecting JSON; return dict or None."""
    hdrs = {
        "User-Agent": "WorldReportBot/1.0 (contact: admin@example.com)",
        "Accept": "application/json",
    }
    if headers:
        hdrs.update(headers)
    try:
        resp = requests.get(url, headers=hdrs, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"JSON GET {url} returned status {resp.status_code}")
    except Exception as e:
        logger.error(f"JSON GET {url} failed: {e}")
    return None


def safe_request_html(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[datetime]]:
    """Perform a GET request expecting HTML; return text and Last-Modified (UTC) if available."""
    hdrs = {
        "User-Agent": "WorldReportBot/1.0 (contact: admin@example.com)",
        "Accept": "text/html",
    }
    if headers:
        hdrs.update(headers)
    try:
        resp = requests.get(url, headers=hdrs, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            lm = resp.headers.get("Last-Modified")
            lm_dt = None
            if lm:
                try:
                    lm_dt = datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S %Z")
                    # Wikipedia returns GMT; enforce timezone UTC
                    lm_dt = lm_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    lm_dt = None
            return resp.text, lm_dt
        logger.warning(f"HTML GET {url} returned status {resp.status_code}")
    except Exception as e:
        logger.error(f"HTML GET {url} failed: {e}")
    return None, None


def normalize_text(text: str) -> str:
    """Clean and normalize text: strip, collapse whitespace, remove citation markers."""
    if not text:
        return ""
    # Remove [number] citation markers and parenthetical citation clutter
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> List[str]:
    """Basic tokenizer using regex; lowercase words, discard punctuation and short tokens."""
    if not text:
        return []
    # Split on non-word characters, filter short tokens, lowercase
    tokens = [t.lower() for t in re.split(r"[^\w]+", text) if len(t) >= 3]
    return tokens


def deterministic_uuid(topic: str, title: str, source_url: str) -> uuid.UUID:
    """Create a deterministic UUID5 based on topic, title, and URL to ensure idempotency."""
    key = f"{topic}|{title}|{source_url}"
    return uuid.uuid5(uuid.NAMESPACE_URL, key)


def parse_sections_from_html(html: str) -> Dict[str, List[str]]:
    """
    Extract sections, paragraphs, and list items from Wikipedia HTML.
    Returns dict with keys: 'title', 'lead', 'sections', 'lists', 'timelines'.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_el = soup.find(id="firstHeading")
    title = title_el.get_text(strip=True) if title_el else ""

    content = soup.find(id="mw-content-text")
    lead = ""
    sections: List[str] = []
    lists: List[str] = []
    timelines: List[str] = []

    if content:
        # Lead paragraphs before first heading
        first_heading = content.find(["h2", "h3"])
        if first_heading:
            lead_paras = first_heading.find_previous_siblings("p")
        else:
            lead_paras = content.find_all("p", recursive=True)
        lead_texts = [normalize_text(p.get_text(" ", strip=True)) for p in lead_paras if p.get_text(strip=True)]
        if lead_texts:
            lead = " ".join(lead_texts[:3])

        # Sections (H2/H3 and their immediate next paragraphs)
        for h in content.find_all(["h2", "h3"], recursive=True):
            sec_title = normalize_text(h.get_text(" ", strip=True))
            # Skip the contents box or references
            if any(k in sec_title.lower() for k in ["contents", "see also", "references", "external links", "notes", "footnotes"]):
                continue
            # Immediate following paragraphs until next heading
            sec_paras = []
            for sib in h.find_next_siblings():
                if sib.name in ["h2", "h3"]:
                    break
                if sib.name == "p":
                    t = normalize_text(sib.get_text(" ", strip=True))
                    if t:
                        sec_paras.append(t)
                if sib.name in ["ul", "ol"]:
                    # add list items to lists
                    for li in sib.find_all("li"):
                        li_text = normalize_text(li.get_text(" ", strip=True))
                        if li_text:
                            lists.append(li_text)
            if sec_paras:
                sections.append(f"{sec_title}: " + " ".join(sec_paras[:2]))

        # Collect timeline-like lists by keyword
        for h in content.find_all(["h2", "h3"], recursive=True):
            sec_title = normalize_text(h.get_text(" ", strip=True)).lower()
            if "timeline" in sec_title or "history" in sec_title or "chronology" in sec_title:
                for sib in h.find_next_siblings():
                    if sib.name in ["h2", "h3"]:
                        break
                    if sib.name in ["ul", "ol"]:
                        for li in sib.find_all("li"):
                            li_text = normalize_text(li.get_text(" ", strip=True))
                            if li_text:
                                timelines.append(li_text)

    return {
        "title": title,
        "lead": lead,
        "sections": sections,
        "lists": lists,
        "timelines": timelines,
    }


def summarize_sections(struct: Dict[str, List[str]]) -> str:
    """Create a concise summary string from extracted sections and lists."""
    parts: List[str] = []
    if struct.get("lead"):
        parts.append(struct["lead"])
    # Add up to 3 sections
    for s in struct.get("sections", [])[:3]:
        parts.append(s)
    # Add up to 5 list bullets merged
    if struct.get("lists"):
        parts.append("Key points: " + "; ".join(struct["lists"][:5]))
    # Add up to 5 timeline entries
    if struct.get("timelines"):
        parts.append("Timeline: " + "; ".join(struct["timelines"][:5]))
    summary = normalize_text(" ".join(parts))
    return summary[:4000]  # bound summary length


def within_window(timestamp: Optional[datetime]) -> bool:
    """Check if a timestamp is within the last 30 days window."""
    if not timestamp:
        return False
    return timestamp >= WINDOW_START


def parse_wiki_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse Wikipedia REST timestamp (ISO8601) to aware UTC datetime."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


# =========================
# Wikipedia scraping utilities
# =========================

def search_wikipedia_pages(query: str, limit: int = 5) -> List[Dict]:
    """Use Wikipedia REST search to find pages; return list of page objects."""
    url = f"{WIKI_API_BASE}/search/page"
    data = safe_request_json(url, params={"q": query, "limit": str(limit)})
    if not data or "pages" not in data:
        return []
    return data["pages"]


def get_page_summary(title: str) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Fetch page summary via REST API.
    Returns (summary_text, canonical_url, timestamp).
    """
    url = f"{WIKI_API_BASE}/page/summary/{quote(title)}"
    data = safe_request_json(url)
    if not data:
        return None, None, None
    summary_text = normalize_text(data.get("extract") or data.get("description") or "")
    canonical_url = data.get("content_urls", {}).get("desktop", {}).get("page") or f"{WIKI_WEB_BASE}/{quote(title)}"
    timestamp = parse_wiki_timestamp(data.get("timestamp"))
    return summary_text, canonical_url, timestamp


def scrape_page_html(title: str) -> Tuple[Optional[Dict[str, List[str]]], Optional[str], Optional[datetime]]:
    """
    Fallback: fetch and parse page HTML.
    Returns (structures_dict, canonical_url, last_modified_timestamp).
    """
    page_url = f"{WIKI_WEB_BASE}/{quote(title)}"
    html, last_modified = safe_request_html(page_url)
    if not html:
        return None, None, None
    struct = parse_sections_from_html(html)
    return struct, page_url, last_modified


def build_item(topic: str, title: str, summary: str, source_url: str, scraped_at: datetime) -> Dict:
    """Create standardized item dict."""
    tokens = simple_tokenize(summary)
    item_id = deterministic_uuid(topic, title, source_url)
    return {
        "id": item_id,
        "topic": topic,
        "title": title,
        "summary": summary,
        "source_url": source_url,
        "scraped_at": scraped_at,
        "tokens": tokens,
    }


def gather_topic_items(topic: str, queries: List[str]) -> List[Dict]:
    """
    Gather items for a given topic:
    - Search via REST API.
    - Use summary first; if no summary or missing sections, fallback to HTML scrape.
    - Respect 30-day window using REST timestamp or Last-Modified header.
    """
    items: List[Dict] = []
    seen_titles: set[str] = set()

    for q in queries:
        pages = search_wikipedia_pages(q, limit=5)
        logger.info(f"Topic '{topic}' query '{q}' returned {len(pages)} pages")
        for p in pages:
            title = p.get("title") or p.get("display_title")
            if not title or title in seen_titles:
                continue

            # First try REST summary
            summary, url, ts = get_page_summary(title)

            if summary and within_window(ts):
                item = build_item(topic, title, summary, url or f"{WIKI_WEB_BASE}/{quote(title)}", NOW_UTC)
                items.append(item)
                seen_titles.add(title)
                continue

            # If no recent summary, fallback to HTML scrape
            struct, html_url, last_modified = scrape_page_html(title)
            if not struct:
                continue

            # Only include if last modified is within window
            if within_window(last_modified):
                summary_text = summarize_sections(struct)
                if not summary_text:
                    continue
                item = build_item(topic, struct.get("title") or title, summary_text, html_url or f"{WIKI_WEB_BASE}/{quote(title)}", NOW_UTC)
                items.append(item)
                seen_titles.add(title)
            else:
                # If neither timestamp is recent, skip
                continue

        # Small delay to be considerate to Wikipedia
        time.sleep(0.7)

        # Cap per topic to avoid overly large PDFs
        if len(items) >= 12:
            break

    return items


# =========================
# PostgreSQL storage layer
# =========================

def get_db_connection():
    """Create a psycopg2 connection using environment variables, with SSL mode."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
    )
    conn.autocommit = False
    return conn


def ensure_schema_and_table(conn) -> None:
    """Create schema and table if they do not exist."""
    create_schema_sql = f"""
    CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};
    """
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.{DB_TABLE} (
        id UUID PRIMARY KEY,
        topic TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        source_url TEXT NOT NULL,
        scraped_at TIMESTAMP WITH TIME ZONE NOT NULL,
        tokens TEXT[] NOT NULL
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_schema_sql)
        cur.execute(create_table_sql)
    conn.commit()
    logger.info(f"Ensured schema '{DB_SCHEMA}' and table '{DB_TABLE}' exist")


def upsert_items(conn, items: List[Dict]) -> Tuple[int, int]:
    """UPSERT items into the database using parameterized queries."""
    if not items:
        return 0, 0

    insert_sql = f"""
    INSERT INTO {DB_SCHEMA}.{DB_TABLE} (id, topic, title, summary, source_url, scraped_at, tokens)
    VALUES %s
    ON CONFLICT (id) DO UPDATE SET
        topic = EXCLUDED.topic,
        title = EXCLUDED.title,
        summary = EXCLUDED.summary,
        source_url = EXCLUDED.source_url,
        scraped_at = EXCLUDED.scraped_at,
        tokens = EXCLUDED.tokens;
    """

    values = [
        (
            str(item["id"]),
            item["topic"],
            item["title"],
            item["summary"],
            item["source_url"],
            item["scraped_at"],
            item["tokens"],
        )
        for item in items
    ]

    inserted = 0
    updated = 0

    # To determine inserted vs updated, check existing IDs first
    existing_ids = set()
    ids = [str(item["id"]) for item in items]
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id::text FROM {DB_SCHEMA}.{DB_TABLE} WHERE id = ANY(%s);",
            (ids,),
        )
        rows = cur.fetchall()
        existing_ids = {r[0] for r in rows}

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values)
    conn.commit()

    for v in values:
        if v[0] in existing_ids:
            updated += 1
        else:
            inserted += 1

    logger.info(f"Database upsert completed: inserted={inserted}, updated={updated}")
    return inserted, updated


# =========================
# PDF builder (reportlab.platypus)
# =========================

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    ListFlowable,
    ListItem,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.pdfbase.pdfmetrics import stringWidth


def build_pdf(items_by_topic: Dict[str, List[Dict]], pdf_path: str) -> None:
    """Generate the executive PDF using reportlab.platypus."""
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title", fontSize=24, leading=28, spaceAfter=18))
    styles.add(ParagraphStyle(name="SubTitle", fontSize=14, leading=18, textColor=colors.gray, spaceAfter=12))
    styles.add(ParagraphStyle(name="TopicHeader", fontSize=18, leading=22, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="ItemTitle", fontSize=12, leading=14, spaceAfter=4, leftIndent=12))
    styles.add(ParagraphStyle(name="Bullet", fontSize=10, leading=14, leftIndent=24))
    styles.add(ParagraphStyle(name="URL", fontSize=8, leading=10, textColor=colors.HexColor("#555555"), leftIndent=24))
    styles.add(ParagraphStyle(name="Footer", fontSize=8, alignment=1, textColor=colors.gray))

    story = []

    # Title page
    title_text = f"State of the World — {NOW_UTC.strftime('%B %Y')}"
    story.append(Paragraph(title_text, styles["Title"]))
    story.append(Paragraph("Executive report compiled from Wikipedia (REST API & HTML).", styles["SubTitle"]))
    story.append(Spacer(1, 0.3 * inch))

    # Table of contents
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontSize=12, name="TOCLevel1", leftIndent=12, firstLineIndent=-6, spaceAfter=6),
    ]
    story.append(Paragraph("Table of contents", styles["TopicHeader"]))
    story.append(toc)
    story.append(PageBreak())

    # Topic sections
    for topic, items in items_by_topic.items():
        # Add topic heading and register with TOC
        topic_para = Paragraph(topic, styles["TopicHeader"])
        # The TOC listens for 'TOCEntry' events via annotations in Paragraph; BaseDocTemplate handles this.
        topic_para._bookmarkName = topic  # anchor
        story.append(topic_para)
        story.append(Spacer(1, 0.1 * inch))

        if not items:
            story.append(Paragraph("No recent items found in the last 30 days.", styles["ItemTitle"]))
            story.append(Spacer(1, 0.2 * inch))
            continue

        # Build bullet list per item
        bullet_items = []
        for item in items:
            title = item["title"]
            summary = item["summary"]
            url = item["source_url"]

            # Item title
            bullet_title = Paragraph(f"• {title}", styles["ItemTitle"])
            bullet_summary = Paragraph(summary, styles["Bullet"])
            bullet_url = Paragraph(url, styles["URL"])

            # Combine into a single flowable group (ListItem for spacing)
            li = ListItem(
                ListFlowable([bullet_title, bullet_summary, bullet_url], bulletType="bullet", start=None),
                leftIndent=0,
                value=None,
            )
            bullet_items.append(li)

        # Render list
        story.append(ListFlowable(bullet_items, bulletType="bullet", start=None))
        story.append(Spacer(1, 0.25 * inch))

    # Page numbering (using platypus PageTemplate onPage callback)
    def _on_page(canvas, doc):
        page_num_text = f"{doc.page}"
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.gray)
        canvas.drawRightString(A4[0] - 0.5 * inch, 0.5 * inch, page_num_text)
        canvas.restoreState()

    # Create document with template
    doc = BaseDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    frame = doc.pageFrames[0] if doc.pageFrames else None
    if frame is None:
        from reportlab.platypus.frames import Frame
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    template = PageTemplate(id="main", frames=[frame], onPage=_on_page)
    doc.addPageTemplates([template])

    # Build document
    doc.build(story)
    logger.info(f"PDF generated at: {pdf_path}")


# =========================
# Monthly execution logic
# =========================

def load_existing_ids(conn, since: datetime) -> set[str]:
    """Load IDs scraped since the monthly window start to avoid duplicates on re-run."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id::text FROM {DB_SCHEMA}.{DB_TABLE} WHERE scraped_at >= %s;",
            (since,),
        )
        rows = cur.fetchall()
    return {r[0] for r in rows}


def main():
    logger.info("Starting monthly State of the World report generation")

    # Database: ensure schema & table
    try:
        conn = get_db_connection()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        ensure_schema_and_table(conn)
    except Exception as e:
        logger.error(f"Schema/table creation failed: {e}")
        conn.close()
        sys.exit(1)

    # Load existing IDs in window to enforce idempotency
    try:
        existing_ids = load_existing_ids(conn, WINDOW_START)
    except Exception as e:
        logger.error(f"Failed to load existing IDs: {e}")
        existing_ids = set()

    # Gather items for each topic
    items_by_topic: Dict[str, List[Dict]] = {}
    total_candidates = 0

    for topic, queries in TOPIC_QUERIES.items():
        try:
            items = gather_topic_items(topic, queries)
            total_candidates += len(items)
            # Filter out already stored items
            new_items = [i for i in items if str(i["id"]) not in existing_ids]
            items_by_topic[topic] = new_items
            logger.info(f"Topic '{topic}': {len(items)} candidates, {len(new_items)} new")
        except Exception as e:
            logger.error(f"Error gathering items for topic '{topic}': {e}")
            items_by_topic[topic] = []

    # Upsert to DB
    all_new_items = [i for items in items_by_topic.values() for i in items]
    inserted = updated = 0
    if all_new_items:
        try:
            inserted, updated = upsert_items(conn, all_new_items)
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            conn.rollback()
            conn.close()
            sys.exit(1)
    else:
        logger.info("No new items to insert")

    # Close DB
    try:
        conn.commit()
        conn.close()
    except Exception:
        pass

    # Build PDF from DB (to ensure canonical data), but we can use in-memory items safely
    try:
        build_pdf(items_by_topic, PDF_PATH)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        sys.exit(1)

    logger.info(f"Report generation complete. Inserted={inserted}, Updated={updated}, PDF={PDF_PATH}")


if __name__ == "__main__":
    # Configure stdout logging for interactive runs, without duplicating handlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(_log_formatter)
        logger.addHandler(sh)
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)
