#!/usr/bin/env python3
"""
state_of_the_world.py
Monthly “State of the World” executive report generator.
Ubuntu 24.04 / Python 3.13+ / PostgreSQL 17+ / reportlab.platypus
Author: <you>
"""

import os
import sys
import re
import uuid
import logging
import logging.handlers
import datetime as dt
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

###############################################################################
# CONSTANTS
###############################################################################
WIKI_API = "https://en.wikipedia.org/api/rest_v1"
WIKI_BASE = "https://en.wikipedia.org"

# Topics we scrape
TOPICS = {
    "emerging_technologies": {
        "search": "Category:Emerging technologies",
        "page": "Emerging technology"
    },
    "political_shifts": {
        "search": "Category:2020s in politics",
        "page": "2020s in political history"
    },
    "corporate_ma": {
        "search": "Category:Mergers and acquisitions",
        "page": "List of largest mergers and acquisitions"
    },
    "industry_trends": {
        "search": "Category:Industry",
        "page": "Industry"
    },
    "nobel_prizes": {
        "search": "Category:Nobel Prizes",
        "page": "Nobel Prize"
    },
    "scientific_breakthroughs": {
        "search": "Category:Scientific discoveries",
        "page": "Breakthrough of the Year"
    }
}

DB_SCHEMA = "world_report"
DB_TABLE = "scraped_items"

LOG_FILE = "/var/log/state_of_the_world.log"
PDF_DIR = os.getenv("PDF_OUTPUT_DIR", "/tmp")
PDF_FILENAME_TMPL = "State_of_the_World_{:%Y-%m}.pdf"

# DB connection env-vars
DB_DSN = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", "5432"),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD"),
}

###############################################################################
# LOGGER SETUP
###############################################################################
def _init_logger() -> logging.Logger:
    log = logging.getLogger("sow")
    log.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(fmt)
    log.addHandler(handler)
    return log


LOG = _init_logger()

###############################################################################
# POSTGRES HELPERS
###############################################################################
class DB:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_DSN)
        self.conn.autocommit = True
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.{DB_TABLE} (
                    id UUID PRIMARY KEY,
                    topic TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    scraped_at TIMESTAMP NOT NULL,
                    tokens TEXT[] NOT NULL,
                    UNIQUE(topic, title)
                )
                """
            )

    def upsert_items(self, items: List[Dict]) -> int:
        if not items:
            return 0
        rows = [
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
        sql = f"""
        INSERT INTO {DB_SCHEMA}.{DB_TABLE}
        (id, topic, title, summary, source_url, scraped_at, tokens)
        VALUES %s
        ON CONFLICT (topic, title)
        DO UPDATE SET
            summary=EXCLUDED.summary,
            source_url=EXCLUDED.source_url,
            scraped_at=EXCLUDED.scraped_at,
            tokens=EXCLUDED.tokens
        """
        with self.conn.cursor() as cur:
            execute_values(cur, sql, rows)
            return len(rows)

    def already_stored(self, topic: str, titles: List[str]) -> set:
        if not titles:
            return set()
        sql = f"""
        SELECT title FROM {DB_SCHEMA}.{DB_TABLE}
        WHERE topic=%s AND title = ANY(%s)
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (topic, titles))
            return {row[0] for row in cur.fetchall()}

###############################################################################
# TOKENIZER
###############################################################################
def tokenize(text: str) -> List[str]:
    # Very small tokenizer: lowercase, split on non-alpha, min length 3
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [tok for tok in text.split() if len(tok) >= 3]
    return list(set(tokens))  # unique

###############################################################################
# WIKIPEDIA HELPERS
###############################################################################
class WikiClient:
    def __init__(self):
        self.sess = requests.Session()
        self.sess.headers.update(
            {"User-Agent": "StateOfTheWorldBot/1.0 (https://example.com)"}
        )

    def get_page_summary(self, title: str) -> Optional[Dict]:
        url = f"{WIKI_API}/page/summary/{title.replace(' ', '_')}"
        resp = self.sess.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return None

    def get_page_html(self, title: str) -> Optional[str]:
        url = f"{WIKI_API}/page/html/{title.replace(' ', '_')}"
        resp = self.sess.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.text
        return None

    def search_titles(self, query: str, limit: int = 20) -> List[str]:
        # Use opensearch
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json",
            "namespace": 0,
        }
        resp = self.sess.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()[1]
        return []

###############################################################################
# SCRAPING STRATEGY
###############################################################################
class Scraper:
    def __init__(self, wiki: WikiClient, db: DB):
        self.wiki = wiki
        self.db = db
        self.now = dt.datetime.utcnow()

    def run(self) -> Dict[str, List[Dict]]:
        collected = {}
        for topic_key, cfg in TOPICS.items():
            LOG.info("Scraping topic: %s", topic_key)
            items = self._scrape_topic(topic_key, cfg)
            new_items = self._filter_new(topic_key, items)
            if new_items:
                self.db.upsert_items(new_items)
            collected[topic_key] = new_items
            LOG.info("Topic %s: %d new items", topic_key, len(new_items))
        return collected

    def _filter_new(self, topic: str, items: List[Dict]) -> List[Dict]:
        titles = [it["title"] for it in items]
        existing = self.db.already_stored(topic, titles)
        return [it for it in items if it["title"] not in existing]

    def _scrape_topic(self, topic_key: str, cfg: Dict) -> List[Dict]:
        # Try primary page first
        page_title = cfg["page"]
        summary = self.wiki.get_page_summary(page_title)
        html = self.wiki.get_page_html(page_title)
        items = []
        if summary and html:
            items.extend(self._parse_page(topic_key, summary, html))
        # If we got nothing, search more
        if not items:
            search_q = cfg["search"]
            more_titles = self.wiki.search_titles(search_q, limit=10)
            for ttl in more_titles:
                summ = self.wiki.get_page_summary(ttl)
                htm = self.wiki.get_page_html(ttl)
                if summ and htm:
                    items.extend(self._parse_page(topic_key, summ, htm))
        return items

    def _parse_page(self, topic_key: str, summary: Dict, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        # Grab first couple of sections
        sections = []
        for sect in soup.find_all("section")[:3]:
            txt = sect.get_text(" ", strip=True)
            if txt:
                sections.append(txt)
        full_text = " ".join(sections)
        tokens = tokenize(full_text)
        return [
            {
                "id": uuid.uuid4(),
                "topic": topic_key,
                "title": summary.get("title", "No title"),
                "summary": summary.get("extract", "")[:500],
                "source_url": WIKI_BASE + "/wiki/" + summary.get("title", "").replace(" ", "_"),
                "scraped_at": self.now,
                "tokens": tokens,
            }
        ]

###############################################################################
# PDF GENERATOR
###############################################################################
class PDFBuilder:
    def __init__(self, data: Dict[str, List[Dict]]):
        self.data = data
        self.styles = getSampleStyleSheet()
        self.styles.add(
            ParagraphStyle(
                name="TOC",
                parent=self.styles["Normal"],
                fontSize=11,
                spaceAfter=6,
            )
        )

    def build(self, filepath: str):
        doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=1 * cm, bottomMargin=1 * cm)
        story = []
        # Title
        title = Paragraph("State of the World — {}".format(dt.date.today().strftime("%B %Y")), self.styles["Title"])
        story.append(title)
        story.append(Spacer(1, 1 * cm))
        # TOC
        story.append(Paragraph("Table of Contents", self.styles["h2"]))
        toc_data = []
        for idx, (topic, items) in enumerate(self.data.items(), 1):
            toc_data.append([Paragraph("{}".format(topic.replace("_", " ").title()), self.styles["TOC"]),
                             Paragraph(str(len(items)), self.styles["TOC"])])
        toc_table = Table(toc_data, colWidths=[12 * cm, 2 * cm])
        toc_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ]
            )
        )
        story.append(toc_table)
        story.append(PageBreak())
        # Sections
        for topic, items in self.data.items():
            story.append(Paragraph(topic.replace("_", " ").title(), self.styles["h1"]))
            if not items:
                story.append(Paragraph("No new items this month.", self.styles["Normal"]))
            for it in items:
                story.append(Paragraph(it["title"], self.styles["h3"]))
                story.append(Paragraph(it["summary"], self.styles["Normal"]))
                story.append(Spacer(1, 6))
                story.append(Paragraph('<a href="{}">source</a>'.format(it["source_url"]), self.styles["Normal"]))
                story.append(Spacer(1, 12))
            story.append(PageBreak())
        doc.build(story)

###############################################################################
# MAIN
###############################################################################
def main():
    LOG.info("=== State of the World run starting ===")
    try:
        wiki = WikiClient()
        db = DB()
        scraper = Scraper(wiki, db)
        new_data = scraper.run()
        # Build PDF
        pdf_path = os.path.join(PDF_DIR, PDF_FILENAME_TMPL.format(dt.date.today()))
        pdf = PDFBuilder(new_data)
        pdf.build(pdf_path)
        LOG.info("PDF generated at %s", pdf_path)
    except Exception as exc:
        LOG.exception("Fatal error: %s", exc)
        sys.exit(1)
    LOG.info("=== Run complete ===")

if __name__ == "__main__":
    main()
