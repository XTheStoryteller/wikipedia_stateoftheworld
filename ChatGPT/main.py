#!/usr/bin/env python3
"""
state_of_world_report.py

Generates a monthly "State of the World" executive PDF report by gathering,
structuring, and storing Wikipedia-derived content in PostgreSQL.

Target environment:
 - Ubuntu 24.04
 - Python 3.13+ (compatible with 3.11+)
 - PostgreSQL 17+ (psycopg2)
 - ReportLab for PDF generation (reportlab.platypus)
 - BeautifulSoup for HTML fallback parsing

Features:
 - Queries Wikipedia (API first; falls back to HTML scraping with BeautifulSoup)
 - Cleans, normalizes and tokenizes text using built-in Python tools
 - Stores structured items in PostgreSQL (schema/world_report, table scraped_items)
 - Idempotent upserts using source_url uniqueness
 - Rotating logging
 - PDF generation using reportlab.platypus (with TOC and page numbers)
 - Safe cron execution (no interactive prompts)
 - Single-file module; run as `python3 state_of_world_report.py`

AUTHOR: Senior Python Engineer (LLM-generated)
"""

import os
import re
import sys
import uuid
import json
import time
import logging
import socket
import requests
import psycopg2
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    TableOfContents,
    PageTemplate,
    Frame,
    NextPageTemplate,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

# ---------------------------
# Constants & Configuration
# ---------------------------

# Topics to query - mapping to useful Wikipedia search terms (tweakable)
TOPICS = {
    "emerging_technologies": "emerging technology OR emerging technologies",
    "political_shifts": "political shift OR political change OR coup OR election",
    "corporate_mna_timelines": "mergers and acquisitions OR acquisition timeline",
    "industry_trends": "industry trends OR market trend",
    "nobel_prizes": "Nobel Prize OR Nobel laureates",
    "scientific_breakthroughs": "scientific breakthrough OR discovery",
}

WIKIPEDIA_API_SEARCH = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_PAGE_URL = "https://en.wikipedia.org/wiki/{title}"

# How many top search results to consider per topic
TOP_N_RESULTS = 3

# Tokenization: minimal token length to keep
MIN_TOKEN_LEN = 2

# PostgreSQL schema and table names
DB_SCHEMA = "world_report"
DB_TABLE = "scraped_items"

# Logging
LOG_DIR = os.getenv("SOW_LOG_DIR", ".")
LOG_FILE = os.path.join(LOG_DIR, "state_of_world.log")
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB
LOG_BACKUP_COUNT = 5

# PDF output
PDF_OUTPUT_DIR = os.getenv("SOW_PDF_DIR", ".")
PDF_FILENAME_TEMPLATE = "State_of_the_World_{asof}.pdf"

# Execution window: "today minus 30 days"
MONTH_WINDOW_DAYS = 30

# Network timeouts
REQUEST_TIMEOUT = 10  # seconds

# ---------------------------
# Setup logging
# ---------------------------

logger = logging.getLogger("state_of_world")
logger.setLevel(logging.INFO)
os.makedirs(LOG_DIR, exist_ok=True)
handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Also print warnings/errors to stderr for immediate visibility in cron logs
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(formatter)
stderr_handler.setLevel(logging.WARNING)
logger.addHandler(stderr_handler)

# ---------------------------
# Utility helpers
# ---------------------------


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_title(title: str) -> str:
    # Create a safe Wikipedia page title slug
    return title.replace(" ", "_")


def safe_request_get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    """Wrapper for requests.get with timeout and basic error handling."""
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        logger.warning("HTTP request failed: %s %s - %s", url, params, e)
        raise


# ---------------------------
# Tokenizer (built-in)
# ---------------------------


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenize text into words using a simple regex-based tokenizer.
    Keeps only tokens of length >= MIN_TOKEN_LEN and lowercases them.
    """
    if not text:
        return []
    tokens = re.findall(r"\b\w+\b", text)
    tokens = [t.lower() for t in tokens if len(t) >= MIN_TOKEN_LEN]
    return tokens


# ---------------------------
# Wikipedia utilities
# ---------------------------


def wiki_search(query: str, limit: int = TOP_N_RESULTS) -> List[str]:
    """
    Use MediaWiki API to search for pages matching the query.
    Returns a list of page titles (not slugs).
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "utf8": 1,
    }
    try:
        r = safe_request_get(WIKIPEDIA_API_SEARCH, params=params, headers={"User-Agent": "StateOfWorldBot/1.0 (contact: none)"})
        data = r.json()
        titles = [item["title"] for item in data.get("query", {}).get("search", [])]
        logger.info("Search query '%s' returned %d titles", query, len(titles))
        return titles
    except Exception as e:
        logger.warning("Wiki search failed for query '%s': %s", query, e)
        return []


def wiki_get_summary(title: str) -> Optional[Dict[str, Any]]:
    """
    Query the REST summary endpoint for a title.
    Returns parsed JSON or None on failure.
    """
    slug = slugify_title(title)
    url = WIKIPEDIA_REST_SUMMARY.format(title=slug)
    try:
        r = safe_request_get(url, headers={"User-Agent": "StateOfWorldBot/1.0 (contact: none)"})
        data = r.json()
        return data
    except Exception as e:
        logger.warning("REST summary failed for '%s': %s", title, e)
        return None


def wiki_scrape_html_sections(title: str) -> Dict[str, Any]:
    """
    Fallback HTML scraping using BeautifulSoup.
    Returns a dict with 'title', 'sections' list where each section is dict(title, paragraphs, lists).
    """
    slug = slugify_title(title)
    url = WIKIPEDIA_PAGE_URL.format(title=slug)
    try:
        r = safe_request_get(url, headers={"User-Agent": "StateOfWorldBot/1.0 (contact: none)"})
        soup = BeautifulSoup(r.text, "html.parser")

        page_title = soup.find(id="firstHeading")
        page_title_text = page_title.get_text(strip=True) if page_title else title

        content_div = soup.find(id="mw-content-text")
        if not content_div:
            logger.warning("No content found on HTML page for '%s'", title)
            return {"title": page_title_text, "sections": []}

        sections = []
        # Wikipedia articles are structured with h2/h3 headers and paragraphs
        for header in content_div.find_all(["h2", "h3"]):
            sec_title = header.get_text(separator=" ", strip=True)
            # Collect sibling elements until the next header of same level
            sec_paragraphs = []
            sec_lists = []
            for sib in header.find_next_siblings():
                if sib.name in ["h2", "h3"]:
                    break
                if sib.name == "p":
                    text = sib.get_text(separator=" ", strip=True)
                    if text:
                        sec_paragraphs.append(text)
                if sib.name in ["ul", "ol"]:
                    # collect list items
                    items = [li.get_text(separator=" ", strip=True) for li in sib.find_all("li")]
                    if items:
                        sec_lists.append(items)
            if sec_paragraphs or sec_lists:
                sections.append({"title": sec_title, "paragraphs": sec_paragraphs, "lists": sec_lists})

        # If no sections found, attempt to collect top paragraphs
        if not sections:
            paras = [p.get_text(separator=" ", strip=True) for p in content_div.find_all("p") if p.get_text(strip=True)]
            if paras:
                sections.append({"title": "Lead", "paragraphs": paras[:3], "lists": []})

        return {"title": page_title_text, "sections": sections, "source_url": url}
    except Exception as e:
        logger.warning("HTML scraping failed for '%s': %s", title, e)
        return {"title": title, "sections": []}


def extract_structured_item_from_title(title: str, topic_key: str) -> Optional[Dict[str, Any]]:
    """
    For a given Wikipedia page title, attempt to fetch a structured item:
    - Use REST summary endpoint first
    - If insufficient, fallback to HTML scraping for sections and paragraphs
    Returns a dict with required fields, or None on failure.
    """
    scraped_at = datetime.now(timezone.utc)
    # Attempt REST first
    summary = wiki_get_summary(title)
    source_url = None
    content_summary = ""
    sections = []

    if summary:
        # REST provides extract, description, titles and content URLs
        source_url = summary.get("content_urls", {}).get("desktop", {}).get("page") or WIKIPEDIA_PAGE_URL.format(title=slugify_title(title))
        # Prefer 'extract' for summary
        content_summary = summary.get("extract", "")
        # Some summaries include sections in 'sections' or may not; if not, fallback
        # The REST summary seldom includes full sections; so we'll note summary and perhaps fallback
        # We'll still attempt to create a compact structured representation:
        sections = []
        if content_summary:
            # split into small paragraphs
            paragraphs = [p.strip() for p in content_summary.split("\n") if p.strip()]
            sections.append({"title": "Summary", "paragraphs": paragraphs, "lists": []})

    # If insufficient content (very short), fallback to HTML scraping
    if not content_summary or len(content_summary.split()) < 30:
        html_data = wiki_scrape_html_sections(title)
        source_url = html_data.get("source_url", source_url or WIKIPEDIA_PAGE_URL.format(title=slugify_title(title)))
        # Build a summary from first few paragraphs
        combined_paras = []
        for sec in html_data.get("sections", []):
            combined_paras.extend(sec.get("paragraphs", [])[:2])
            if len(combined_paras) >= 3:
                break
        content_summary = " ".join(combined_paras[:3]).strip()
        sections = html_data.get("sections", [])

    if not content_summary:
        logger.info("Skipping '%s' - no usable content extracted", title)
        return None

    # Clean/normalize text
    content_summary = clean_text(content_summary)
    # Tokenize
    tokens = simple_tokenize(content_summary)

    item = {
        "id": str(uuid.uuid4()),
        "topic": topic_key,
        "title": title,
        "summary": content_summary,
        "source_url": source_url or WIKIPEDIA_PAGE_URL.format(title=slugify_title(title)),
        "scraped_at": scraped_at.replace(microsecond=0).isoformat(),
        "tokens": tokens,
        "sections": sections,  # included for PDF rendering but not stored in DB
    }
    return item


def clean_text(text: str) -> str:
    """
    Basic cleaning: normalize whitespace, remove references like [1], and strip.
    """
    # Remove citation brackets [1], [2], [a], etc.
    text = re.sub(r"\[[^\]]+\]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------
# Database layer (psycopg2)
# ---------------------------


def get_db_conn():
    """
    Create a psycopg2 connection using environment variables:
     - DB_HOST
     - DB_PORT
     - DB_NAME
     - DB_USER
     - DB_PASSWORD
     - DB_SSLMODE (optional)
    """
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_sslmode = os.getenv("DB_SSLMODE", None)

    conn_params = {
        "host": db_host,
        "port": db_port,
        "dbname": db_name,
        "user": db_user,
        "password": db_password,
    }
    if db_sslmode:
        conn_params["sslmode"] = db_sslmode

    # psycopg2 uses a connection string; supply parameters
    conn_str = " ".join([f"{k}={v}" for k, v in conn_params.items() if v is not None])
    try:
        conn = psycopg2.connect(conn_str)
        conn.autocommit = False
        return conn
    except Exception as e:
        logger.exception("Unable to connect to PostgreSQL: %s", e)
        raise


def ensure_schema_and_table(conn: psycopg2.extensions.connection):
    """
    Create schema and table (and a unique index on source_url) if not exist.
    Uses parameterized statements where necessary.
    """
    create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};"
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.{DB_TABLE} (
        id UUID PRIMARY KEY,
        topic TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT,
        source_url TEXT NOT NULL,
        scraped_at TIMESTAMP WITH TIME ZONE,
        tokens TEXT[]
    );
    """
    create_unique_index_sql = f"""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = %s AND indexname = %s
        ) THEN
            CREATE UNIQUE INDEX {DB_TABLE}_source_url_key ON {DB_SCHEMA}.{DB_TABLE} (source_url);
        END IF;
    END;
    $$;
    """
    cur = conn.cursor()
    try:
        cur.execute(create_schema_sql)
        cur.execute(create_table_sql)
        cur.execute(create_unique_index_sql, (DB_SCHEMA, DB_TABLE, f"{DB_TABLE}_source_url_key"))
        conn.commit()
        logger.info("Ensured schema %s and table %s exist", DB_SCHEMA, DB_TABLE)
    except Exception:
        conn.rollback()
        logger.exception("Failed to ensure schema/table; rolling back")
        raise
    finally:
        cur.close()


def upsert_item(conn: psycopg2.extensions.connection, item: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Insert or update an item (upsert by source_url).
    Returns (inserted, updated) booleans.
    """
    upsert_sql = f"""
    INSERT INTO {DB_SCHEMA}.{DB_TABLE} (id, topic, title, summary, source_url, scraped_at, tokens)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (source_url)
    DO UPDATE SET
        topic = EXCLUDED.topic,
        title = EXCLUDED.title,
        summary = EXCLUDED.summary,
        scraped_at = EXCLUDED.scraped_at,
        tokens = EXCLUDED.tokens
    RETURNING (xmax = 0) AS inserted; -- True if newly inserted (xmax == 0)
    """
    cur = conn.cursor()
    try:
        scraped_at_val = datetime.fromisoformat(item["scraped_at"])
        cur.execute(
            upsert_sql,
            (
                item["id"],
                item["topic"],
                item["title"],
                item["summary"],
                item["source_url"],
                scraped_at_val,
                item["tokens"],
            ),
        )
        # RETURNING ... yields a row with boolean inserted
        row = cur.fetchone()
        conn.commit()
        inserted = bool(row[0]) if row is not None else True
        updated = not inserted
        if inserted:
            logger.info("Inserted item: %s (%s)", item["title"], item["source_url"])
        else:
            logger.info("Updated item: %s (%s)", item["title"], item["source_url"])
        return inserted, updated
    except Exception:
        conn.rollback()
        logger.exception("Failed to upsert item: %s", item.get("title"))
        raise
    finally:
        cur.close()


def fetch_existing_source_urls(conn: psycopg2.extensions.connection) -> set:
    """
    Fetch all source_url values currently in DB to avoid re-fetching/adding duplicates.
    Returns a set of URLs.
    """
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT source_url FROM {DB_SCHEMA}.{DB_TABLE};")
        rows = cur.fetchall()
        return set(r[0] for r in rows if r and r[0])
    except Exception:
        logger.exception("Failed to fetch existing source URLs")
        return set()
    finally:
        cur.close()


# ---------------------------
# PDF generation (reportlab.platypus)
# ---------------------------


class NumberedCanvasTemplate:
    """
    Helper to add page numbering and header/footer using PageTemplates + canvas callbacks.
    """

    @staticmethod
    def add_page_number(canvas_obj, doc):
        page_num = canvas_obj.getPageNumber()
        text = f"Page {page_num}"
        canvas_obj.setFont("Helvetica", 8)
        width, height = A4
        canvas_obj.drawRightString(width - 20 * mm, 10 * mm, text)

    @staticmethod
    def add_title_header(canvas_obj, doc, title_text):
        canvas_obj.setFont("Helvetica-Bold", 12)
        width, height = A4
        canvas_obj.drawCentredString(width / 2.0, height - 15 * mm, title_text)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawRightString(width - 20 * mm, height - 20 * mm, datetime.now().strftime("%Y-%m-%d"))


def build_pdf_report(items_by_topic: Dict[str, List[Dict[str, Any]]], asof: datetime, output_path: str):
    """
    Build a PDF report using reportlab.platypus.
    - items_by_topic: mapping topic_key -> list of item dicts (with title, summary, source_url, sections)
    - asof: report date
    - output_path: file path to write the PDF
    """
    logger.info("Building PDF report at %s", output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Document setup
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=30 * mm, bottomMargin=20 * mm)

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, spaceAfter=12)
    subtitle_style = ParagraphStyle("SubtitleStyle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10, spaceAfter=18)
    h1_style = ParagraphStyle("H1", parent=styles["Heading1"], alignment=TA_LEFT, fontSize=16, spaceBefore=12, spaceAfter=6)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], alignment=TA_LEFT, fontSize=12, spaceBefore=8, spaceAfter=4)
    normal_style = ParagraphStyle("Normal", parent=styles["BodyText"], fontSize=10, spaceAfter=6, leading=12)
    small_style = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=10)

    story = []

    # Title page
    title_text = "State of the World — Executive Report"
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph(f"As of {asof.strftime('%Y-%m-%d')}", subtitle_style))
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("Generated by StateOfWorldBot", normal_style))
    story.append(PageBreak())

    # Table of contents
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontSize=12, name="TOCHeading1", leftIndent=12, firstLineIndent=-12, spaceBefore=6, leading=14),
        ParagraphStyle(fontSize=10, name="TOCHeading2", leftIndent=24, firstLineIndent=-12, spaceBefore=2, leading=12),
    ]
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(Spacer(1, 6))
    story.append(toc)
    story.append(PageBreak())

    # For TOC registration, use nested flowables and keepTogether where helpful
    for topic_key, items in items_by_topic.items():
        # Human-friendly topic title
        topic_title = topic_key.replace("_", " ").title()
        story.append(Paragraph(topic_title, h1_style))
        # Register TOC entry
        # TableOfContents listens for 'TOCEntry' notifications; Paragraphs with style 'Heading1' generate them automatically.
        # But to be safe we can add a heading paragraph with the h1_style (which is a Heading1)
        # Each item will be a subheading (h2_style)
        story.append(Spacer(1, 4))

        if not items:
            story.append(Paragraph("No notable items found this period.", normal_style))
            story.append(PageBreak())
            continue

        for item in items:
            # Item heading
            story.append(Paragraph(item["title"], h2_style))
            # Summary as bullets (brief)
            # We'll split the summary into sentences and present top 3 as bullets
            summary = item.get("summary", "")
            bullets = split_into_sentences(summary)[:5]
            for b in bullets:
                # Use a simple dash as bullet
                story.append(Paragraph(f"• {b}", normal_style))
            # Small source URL
            story.append(Paragraph(f"Source: {item.get('source_url')}", small_style))
            story.append(Spacer(1, 6))

        story.append(PageBreak())

    # Build the PDF with page templates to add page numbers/headers
    def on_first_page(canvas_obj, doc_obj):
        NumberedCanvasTemplate.add_title_header(canvas_obj, doc_obj, title_text)
        NumberedCanvasTemplate.add_page_number(canvas_obj, doc_obj)

    def on_later_pages(canvas_obj, doc_obj):
        NumberedCanvasTemplate.add_title_header(canvas_obj, doc_obj, title_text)
        NumberedCanvasTemplate.add_page_number(canvas_obj, doc_obj)

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    logger.info("PDF generation completed: %s", output_path)


def split_into_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter. Not perfect but avoids external dependencies.
    """
    if not text:
        return []
    # Split on ., !, ? followed by space and capital letter or end of string.
    sentences = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# ---------------------------
# Monthly execution logic
# ---------------------------


def collect_topic_items_for_window(topic_key: str, window_days: int = MONTH_WINDOW_DAYS) -> List[Dict[str, Any]]:
    """
    For a given topic_key, perform wiki search, fetch top results, extract items,
    and return items that were 'scraped' within the monthly window (we stamp with now).
    We still return items but the DB insertion logic will avoid duplicates.
    """
    query = TOPICS.get(topic_key, topic_key)
    titles = wiki_search(query, limit=TOP_N_RESULTS)
    collected = []
    for t in titles:
        # For each title, extract structured item
        try:
            item = extract_structured_item_from_title(t, topic_key)
            if item:
                # scraped_at is now; we include all items but later the script logic can filter by date if needed
                collected.append(item)
            # be polite to wikipedia
            time.sleep(0.5)
        except Exception as e:
            logger.warning("Failed to collect item for title '%s': %s", t, e)
            continue
    logger.info("Collected %d items for topic %s", len(collected), topic_key)
    return collected


def run_monthly_collection_and_report():
    """
    Main orchestration:
     - Compute monthly window
     - Connect to DB and ensure schema/table
     - For each topic, collect items, upsert to DB (idempotent)
     - Generate PDF compiled from collected items in this run + optionally existing DB items
    """
    logger.info("Starting monthly collection run")
    asof = datetime.now(timezone.utc)
    window_start = asof - timedelta(days=MONTH_WINDOW_DAYS)
    logger.info("Monthly window: %s to %s", window_start.isoformat(), asof.isoformat())

    # Connect to DB
    conn = get_db_conn()
    try:
        ensure_schema_and_table(conn)
    except Exception as e:
        logger.exception("Database preparation failed: %s", e)
        conn.close()
        return

    # Fetch existing source URLs to avoid duplicates
    try:
        existing_urls = fetch_existing_source_urls(conn)
    except Exception:
        existing_urls = set()

    # Collect items by topic
    items_by_topic = {}
    for topic_key in TOPICS.keys():
        try:
            collected = collect_topic_items_for_window(topic_key)
        except Exception as e:
            logger.exception("Failed to collect for topic %s: %s", topic_key, e)
            collected = []
        # Only keep those not already present (by source_url)
        new_items = []
        for it in collected:
            if it["source_url"] in existing_urls:
                logger.info("Skipping already-stored source: %s", it["source_url"])
                continue
            # Basic filter: exclude trivial summaries
            if len(it.get("tokens", [])) < 5:
                logger.info("Skipping short item: %s", it.get("title"))
                continue
            # Upsert into DB
            try:
                inserted, updated = upsert_item(conn, it)
                if inserted:
                    new_items.append(it)
                else:
                    # if updated, we may include it in report too
                    new_items.append(it)
                    existing_urls.add(it["source_url"])
            except Exception:
                logger.exception("Upsert failed for item %s", it.get("title"))
                continue

        # Add any existing DB items for this topic within the window for richer report
        # We'll fetch from DB: titles and summaries where scraped_at >= window_start and topic matches
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT id, topic, title, summary, source_url, scraped_at FROM {DB_SCHEMA}.{DB_TABLE} WHERE topic = %s AND scraped_at >= %s ORDER BY scraped_at DESC;",
                (topic_key, window_start),
            )
            rows = cur.fetchall()
            db_items = []
            for r in rows:
                db_items.append(
                    {
                        "id": str(r[0]),
                        "topic": r[1],
                        "title": r[2],
                        "summary": r[3],
                        "source_url": r[4],
                        "scraped_at": r[5].isoformat() if r[5] else None,
                    }
                )
            cur.close()
        except Exception:
            logger.exception("Failed to fetch DB items for topic %s", topic_key)
            db_items = []

        # Merge new_items and db_items, with preference to new_items first
        combined = new_items + [d for d in db_items if d["source_url"] not in {ni["source_url"] for ni in new_items}]
        items_by_topic[topic_key] = combined

    # Build PDF using items_by_topic
    pdf_name = PDF_FILENAME_TEMPLATE.format(asof=asof.strftime("%Y%m%d"))
    pdf_path = os.path.join(PDF_OUTPUT_DIR, pdf_name)
    try:
        build_pdf_report(items_by_topic, asof, pdf_path)
    except Exception:
        logger.exception("PDF generation failed")
    finally:
        conn.close()
        logger.info("Database connection closed")

    logger.info("Monthly run complete. PDF saved to %s", pdf_path)


# ---------------------------
# Entry point
# ---------------------------

def main():
    """
    Run the monthly collection and report generation.
    Designed to be safe for cron execution (non-interactive).
    """
    # Basic environment check: ensure connectivity to the internet/resolvable hosts
    try:
        # Simple DNS check to wikipedia.org
        socket.gethostbyname("en.wikipedia.org")
    except Exception:
        logger.warning("DNS resolution for en.wikipedia.org failed - aborting run")
        return

    try:
        run_monthly_collection_and_report()
    except Exception as e:
        logger.exception("Unhandled exception in main(): %s", e)
        # Do not re-raise when running under cron; log and exit with non-zero code
        sys.exit(1)


if __name__ == "__main__":
    main()


# ---------------------------
# Shell instructions (README-style)
# ---------------------------
#
# Save this file as `state_of_world_report.py`.
#
# 1) Create a Python virtual environment and install dependencies:
#
#    sudo apt update
#    sudo apt install -y python3.13 python3.13-venv python3-pip libpq-dev build-essential
#
#    # Create venv (adjust python3.13 to python3 if that's the default)
#    python3.13 -m venv venv
#    source venv/bin/activate
#
#    pip install --upgrade pip
#    pip install requests beautifulsoup4 psycopg2-binary reportlab
#
# 2) PostgreSQL setup (run as a user with privileges, e.g., postgres):
#
#    # Connect to postgres and create a database/user as needed:
#    sudo -u postgres psql
#    -- inside psql:
#    CREATE DATABASE worldreports;
#    CREATE USER sow_user WITH PASSWORD 'strong_password_here';
#    GRANT ALL PRIVILEGES ON DATABASE worldreports TO sow_user;
#    \q
#
#    # The script will create schema/world_report and required table automatically when run.
#
# 3) Set environment variables for DB connection and output paths:
#
#    # Example (export these in the environment where cron runs):
#    export DB_HOST=localhost
#    export DB_PORT=5432
#    export DB_NAME=worldreports
#    export DB_USER=sow_user
#    export DB_PASSWORD='strong_password_here'
#    # Optional:
#    export DB_SSLMODE=disable
#    export SOW_LOG_DIR=/var/log/state_of_world
#    export SOW_PDF_DIR=/var/reports/state_of_world
#
#    # Ensure directories exist and have appropriate permissions:
#    sudo mkdir -p /var/log/state_of_world
#    sudo mkdir -p /var/reports/state_of_world
#    sudo chown $(whoami) /var/log/state_of_world /var/reports/state_of_world
#
# 4) Run the script manually:
#
#    source venv/bin/activate
#    python state_of_world_report.py
#
# 5) Cron scheduling (example: run on 1st of every month at 02:00):
#
#    crontab -e
#    # add:
#    0 2 1 * * /path/to/venv/bin/python /path/to/state_of_world_report.py >> /var/log/state_of_world/cron-run.log 2>&1
#
# Notes & considerations:
# - The script is designed to be idempotent by enforcing a unique index on source_url
#   and performing upserts. Re-running will not create duplicates.
# - The tokenizer is a simple built-in tokenizer. If you require more advanced tokenization
#   (languages, stopwords), consider integrating spaCy or NLTK (not included to keep the
#   dependency list minimal and Ubuntu-friendly).
# - Wikipedia scraping is rate-limited by politeness sleeps; adjust time.sleep if running
#   into API rate limits.
# - Ensure the environment variables are available to the cron job (cron runs with a limited environment).
#   You can wrap the run in a shell script that sets env vars or source a file with exports.
#
# End of file.