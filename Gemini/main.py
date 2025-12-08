import os
import logging
import logging.handlers
import requests
import json
import re
import uuid
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Third-party libraries
try:
    from bs4 import BeautifulSoup
    import psycopg2
    from psycopg2 import extras
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import nltk
    from nltk.tokenize import word_tokenize
except ImportError as e:
    print(f"Required library missing: {e}")
    print("Please install dependencies: pip install requests beautifulsoup4 psycopg2-binary reportlab nltk")
    sys.exit(1)

# Ensure NLTK data for tokenization is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# --- Constants ---

REPORT_TITLE = "State of the World Executive Report"
LOG_FILE = "world_report.log"
PDF_FILE = "State_of_the_World_Report.pdf"
DB_SCHEMA = "world_report"
REPORT_WINDOW_DAYS = 30
WIKI_API_URL = "https://en.wikipedia.org/w/rest.php/v1/page/summary/{title}"
WIKI_BASE_URL = "https://en.wikipedia.org"

# Topics and their primary Wikipedia search titles/strategies
# NOTE: These titles are challenging and require fine-tuning for a real-world production system.
# The current selection aims to use broad, recent, or annual pages for demonstration.
TOPICS: Dict[str, str] = {
    "Emerging technologies": "Timeline of greatest inventions", # A good scraping target
    "Political shifts": "List of sovereign states by date of formation", # Good structured list
    "Corporate M&A timelines": "List of largest mergers and acquisitions", # Good structured list/table
    "Industry trends": "Technological and industrial history", # Requires deep scraping
    "Nobel Prizes": f"List of Nobel laureates by country", # Consistent annual data
    "Scientific breakthroughs": "List of years in science", # Annual pages for recent data
}

# --- Logging Setup ---

def setup_logging():
    """Configures rotating file and console logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10485760, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s: %(message)s')
    )
    logger.addHandler(console_handler)
    return logger

LOGGER = setup_logging()


# --- Database Functions ---

class DBManager:
    """Manages secure PostgreSQL connection and operations."""
    def __init__(self):
        self.conn_params = self._load_db_params()

    def _load_db_params(self):
        """Loads DB parameters from environment variables."""
        try:
            return {
                "host": os.environ["PGHOST"],
                "database": os.environ["PGDATABASE"],
                "user": os.environ["PGUSER"],
                "password": os.environ["PGPASSWORD"],
                "port": os.environ.get("PGPORT", "5432")
            }
        except KeyError as e:
            LOGGER.error(f"Missing PostgreSQL environment variable: {e}")
            raise

    def get_connection(self):
        """Establishes and returns a PostgreSQL connection."""
        try:
            return psycopg2.connect(**self.conn_params)
        except psycopg2.Error as e:
            LOGGER.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def setup_schema(self):
        """Creates the schema and table if they don't exist."""
        sql_schema = f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};"
        sql_table = f"""
        CREATE TABLE IF NOT EXISTS {DB_SCHEMA}.scraped_items (
            id UUID PRIMARY KEY,
            topic TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT,
            source_url TEXT,
            scraped_at TIMESTAMP NOT NULL,
            tokens TEXT[]
        );
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_schema)
                    cur.execute(sql_table)
                    conn.commit()
            LOGGER.info(f"Database schema '{DB_SCHEMA}' and table 'scraped_items' ensured.")
        except psycopg2.Error as e:
            LOGGER.error(f"Error setting up database schema: {e}")
            raise

    def upsert_items(self, items: List[Dict[str, Any]]):
        """Inserts or updates items using parameterized queries and UPSERT (ON CONFLICT)."""
        if not items:
            LOGGER.info("No items to upsert.")
            return

        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # The conflict target should be a unique constraint, but since we are generating
            # a new UUID for every scrape, we will use the `title` and `topic` as a 
            # proxy for idempotency (though this is a simplification for Wikipedia data).
            # True idempotency would require checking if (title, topic) exists before generating a new ID.
            # We'll stick to a simple INSERT and assume the calling logic handles uniqueness by title/topic 
            # to avoid generating new UUIDs for existing content.
            
            # Instead of a complex UPSERT, we will use `psycopg2.extras.execute_batch` for performance
            # and rely on the calling logic to filter out known items by title/topic.
            # However, since the requirement is to use UPSERT logic to avoid duplicates,
            # we must check for existence first or use a unique constraint on (topic, title).
            # For this demo, let's create the unique constraint and use proper UPSERT.
            
            # Ensure unique constraint on (topic, title)
            cur.execute(f"""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'scraped_items_topic_title_key'
                    ) THEN
                        ALTER TABLE {DB_SCHEMA}.scraped_items ADD CONSTRAINT scraped_items_topic_title_key UNIQUE (topic, title);
                    END IF;
                END $$;
            """)
            
            upsert_query = f"""
            INSERT INTO {DB_SCHEMA}.scraped_items (id, topic, title, summary, source_url, scraped_at, tokens)
            VALUES (%(id)s, %(topic)s, %(title)s, %(summary)s, %(source_url)s, %(scraped_at)s, %(tokens)s)
            ON CONFLICT (topic, title) DO UPDATE
            SET
                summary = EXCLUDED.summary,
                source_url = EXCLUDED.source_url,
                scraped_at = EXCLUDED.scraped_at,
                tokens = EXCLUDED.tokens
            WHERE
                {DB_SCHEMA}.scraped_items.scraped_at < EXCLUDED.scraped_at;
            """
            
            extras.execute_batch(cur, upsert_query, items)
            conn.commit()
            LOGGER.info(f"Successfully upserted {len(items)} items.")

        except psycopg2.Error as e:
            LOGGER.error(f"Database upsert error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def fetch_report_data(self, window_start: datetime) -> List[Dict[str, Any]]:
        """Fetches all data for the report, focusing on recent entries."""
        query = f"""
        SELECT topic, title, summary, source_url, scraped_at
        FROM {DB_SCHEMA}.scraped_items
        WHERE scraped_at >= %s
        ORDER BY topic, scraped_at DESC;
        """
        data = []
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(query, (window_start,))
                for row in cur:
                    # Convert to Dict for consistent return type
                    data.append(dict(row))
            LOGGER.info(f"Fetched {len(data)} rows for the report.")
        except psycopg2.Error as e:
            LOGGER.error(f"Database fetch error: {e}")
            raise
        finally:
            if conn:
                conn.close()
        return data


# --- Data Utilities ---

def tokenizer(text: str) -> List[str]:
    """Tokenizes text using NLTK's word_tokenize."""
    if not text:
        return []
    # Simple cleanup: remove special characters and lower case
    clean_text = re.sub(r'[^A-Za-z0-9\s]+', '', text).lower()
    return word_tokenize(clean_text)

def clean_text(text: str) -> str:
    """Basic text cleaning to remove multiple spaces and non-printable chars."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Wikipedia Scraper Utilities ---

def get_wiki_page_summary(title: str) -> Optional[Dict[str, Any]]:
    """Fetches a page summary using the Wikipedia REST API."""
    url = WIKI_API_URL.format(title=title)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check if the page exists and has a summary
        if 'title' in data and 'extract' in data:
            return {
                "title": data.get('title'),
                "summary": data.get('extract'),
                "content_urls": data.get('content_urls', {}).get('desktop', {}).get('page')
            }
        return None
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            LOGGER.warning(f"Wikipedia API page not found for '{title}'. Falling back to scraping.")
            return None
        LOGGER.error(f"Wikipedia API HTTP Error for '{title}': {e}")
        return None
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Wikipedia API Request Error for '{title}': {e}")
        return None

def scrape_wiki_page(title: str) -> Optional[Dict[str, Any]]:
    """Scrapes a full Wikipedia page for structured data."""
    url = f"{WIKI_BASE_URL}/wiki/{title}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Basic extraction logic: Look for key sections/lists/paragraphs
        
        # Get the main content div
        content = soup.find(id="mw-content-text")
        if not content:
            LOGGER.warning(f"Could not find main content for scraping page '{title}'.")
            return None
            
        # Example of targeted scraping: find all sections (h2, h3, etc.) and their following content
        scraped_data = []
        
        # We'll target the first few paragraphs and any bulleted/numbered lists
        
        # 1. Main body paragraphs (up to a limit)
        paragraphs = content.find_all('p', limit=5)
        for p in paragraphs:
            text = clean_text(p.get_text())
            if text and not text.startswith("Coordinates"): # Skip coordinate lines
                scraped_data.append(text)

        # 2. Lists (ul/ol)
        lists = content.find_all(['ul', 'ol'], limit=3)
        for list_tag in lists:
            list_items = [clean_text(li.get_text()) for li in list_tag.find_all('li')]
            if list_items:
                scraped_data.extend(list_items[:5]) # Take top 5 items from each list

        full_text = ' '.join(scraped_data)
        
        if not full_text:
            return None

        return {
            "title": soup.find('h1', id='firstHeading').get_text() if soup.find('h1', id='firstHeading') else title,
            "summary": full_text,
            "source_url": url
        }
    
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Wikipedia Scraping Error for '{title}': {e}")
        return None

def process_topic(topic: str, wiki_title: str) -> List[Dict[str, Any]]:
    """
    Attempts to fetch data first via API, then via scraping.
    Constructs the final structured data objects.
    """
    LOGGER.info(f"--- Processing topic: {topic} ---")
    
    # 1. Try API first
    api_data = get_wiki_page_summary(wiki_title)
    
    # 2. Fallback to scraping if API fails or lacks detail
    if api_data:
        LOGGER.info(f"Successfully retrieved API summary for '{wiki_title}'.")
        data_source = api_data
    else:
        LOGGER.info(f"Falling back to scraping for '{wiki_title}'.")
        scraped_data = scrape_wiki_page(wiki_title)
        if scraped_data:
            data_source = scraped_data
        else:
            LOGGER.warning(f"Failed to get any data for '{wiki_title}'. Skipping topic.")
            return []

    # 3. Structure and tokenize the content
    
    # In a real system, 'summary' would be broken down into multiple items based on
    # structured content (lists, tables, sections) found during scraping.
    # For this demonstration, we'll create a single main item per topic.
    
    summary = clean_text(data_source.get("summary", "No summary available."))
    tokens = tokenizer(summary)
    
    item = {
        "id": uuid.uuid4(),
        "topic": topic,
        "title": clean_text(data_source.get("title", wiki_title)),
        "summary": summary,
        "source_url": data_source.get("source_url", f"{WIKI_BASE_URL}/wiki/{wiki_title}"),
        "scraped_at": datetime.now(),
        "tokens": tokens # PostgreSQL requires a list-like object for TEXT[]
    }
    
    # Return as a list of items (even if only one)
    return [item]


# --- PDF Generation Functions ---

class PDFReportGenerator:
    """Generates a professional PDF executive report using ReportLab."""
    
    def __init__(self, filename: str, report_date: datetime):
        self.filename = filename
        self.report_date_str = report_date.strftime("%B %d, %Y")
        self.styles = getSampleStyleSheet()
        self.elements: List[Any] = []
        self.doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=inch/2,
            leftMargin=inch/2,
            topMargin=inch/2,
            bottomMargin=inch/2
        )
        self.page_layout = [] # Used for TOC
        
        # Custom Styles
        self.styles.add(ParagraphStyle(name='TitlePageTitle', fontSize=32, spaceAfter=24, alignment=1))
        self.styles.add(ParagraphStyle(name='TitlePageDate', fontSize=18, spaceAfter=100, alignment=1))
        self.styles.add(ParagraphStyle(name='Heading1', fontSize=18, spaceBefore=20, spaceAfter=10))
        self.styles.add(ParagraphStyle(name='Heading2', fontSize=14, spaceBefore=10, spaceAfter=5, textColor=colors.navy))
        self.styles.add(ParagraphStyle(name='Bullet', fontSize=10, spaceBefore=5, leftIndent=0.5*inch, firstLineIndent=0, bulletIndent=0))
        self.styles.add(ParagraphStyle(name='SourceURL', fontSize=8, textColor=colors.blue, spaceBefore=3, spaceAfter=15))

    def _page_template(self, canvas, doc):
        """Standard page number and header template."""
        canvas.saveState()
        
        # Footer: Page number
        p_text = f"Page {canvas.getPageNumber()}"
        canvas.setFont('Helvetica', 9)
        canvas.drawCentredString(letter[0]/2, 0.35 * inch, p_text)
        
        # Header: Report Title
        canvas.setFont('Helvetica-Bold', 10)
        canvas.drawString(0.5 * inch, letter[1] - 0.35 * inch, REPORT_TITLE)
        
        canvas.restoreState()

    def _build_title_page(self):
        """Adds the main title page."""
        self.elements.append(Spacer(1, 2 * inch))
        self.elements.append(Paragraph(REPORT_TITLE, self.styles['TitlePageTitle']))
        self.elements.append(Paragraph(f"Generated on {self.report_date_str}", self.styles['TitlePageDate']))
        self.elements.append(Paragraph("Prepared by Python Automation Engine", self.styles['Heading2']))
        self.elements.append(PageBreak())

    def _build_toc(self, story):
        """Generates a table of contents."""
        self.elements.append(Paragraph("Table of Contents", self.styles['Heading1']))
        self.elements.append(Spacer(1, 0.2*inch))
        
        toc_items = []
        # Populate TOC items from the story, which now contains a list of (level, text, page)
        for level, text, page_num in self.page_layout:
            if level == 1: # Only include main section headings
                toc_items.append(
                    ListItem(
                        Paragraph(f"{text} . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . {page_num}", self.styles['Bullet']),
                        leftIndent=0,
                        bulletIndent=0
                    )
                )

        self.elements.append(ListFlowable(
            toc_items,
            bulletType='none',
            start=0,
            leftIndent=0.2 * inch,
            spaceBefore=0,
            spaceAfter=0
        ))
        self.elements.append(PageBreak())

    def generate_report(self, data: List[Dict[str, Any]]):
        """Main method to assemble the PDF."""
        self._build_title_page()
        
        # The TOC must be placed after data structure but before the final build, 
        # so we'll build the content first and insert the TOC later.
        content_elements: List[Any] = []
        
        # Group data by topic
        data_by_topic: Dict[str, List[Dict[str, Any]]] = {}
        for item in data:
            data_by_topic.setdefault(item['topic'], []).append(item)

        topic_counter = 0
        for topic in TOPICS.keys(): # Use the official topic order
            if topic in data_by_topic:
                topic_counter += 1
                
                # Add main topic heading
                title_text = f"Section {topic_counter}: {topic}"
                content_elements.append(Paragraph(title_text, self.styles['Heading1'], namedAttr='Heading1'))
                self.page_layout.append((1, title_text, -1)) # -1 is placeholder for page num
                content_elements.append(Spacer(1, 0.1 * inch))
                
                # Add individual items as bullet points
                for item in data_by_topic[topic]:
                    # Sub-heading: Item Title
                    content_elements.append(Paragraph(item['title'], self.styles['Heading2']))
                    
                    # Bullet Summary (break summary into a list of sentences/paragraphs if possible)
                    # Simple bullet point of the entire summary
                    bullet_list_items = [
                        ListItem(
                            Paragraph(clean_text(item['summary']), self.styles['Bullet'], bulletText='•'), 
                            leftIndent=0
                        )
                    ]
                    
                    content_elements.append(ListFlowable(
                        bullet_list_items,
                        bulletType='bullet',
                        start='bullet',
                        leftIndent=0.5 * inch,
                        spaceBefore=0,
                        spaceAfter=0
                    ))
                    
                    # Source URL in small text
                    content_elements.append(Paragraph(f"Source: {item['source_url']}", self.styles['SourceURL']))
                    content_elements.append(Spacer(1, 0.1 * inch))
                
                content_elements.append(PageBreak())
        
        # Now, insert the TOC between the Title Page and Content
        self.elements.extend(content_elements)

        # Build the document, passing the page numbering function
        # The TOC will be generated *during* the build process
        self.doc.build(
            self.elements, 
            onFirstPage=self._page_template, 
            onLaterPages=self._page_template,
            canvasmaker=TableOfContentsBuilder(self.page_layout)
        )
        
        LOGGER.info(f"PDF Report successfully generated: {self.filename}")


# Helper class to build a dummy TOC structure during the build process
class TableOfContentsBuilder:
    def __init__(self, page_layout):
        self.page_layout = page_layout

    def __call__(self, canvas, doc):
        """Called for every page after it has been rendered."""
        
        # Scan the canvas's story for headings and update the page_layout
        for story_index, (level, text, page_num) in enumerate(self.page_layout):
            # Check if the text matches a registered heading on the current page
            # This is a simplification; ReportLab has a built-in mechanism for TOC, 
            # but using platypus elements directly requires this manual tracking.
            if page_num == -1: # Only track unassigned pages
                # A heuristic: if the first element on the page is a known heading
                if doc.page_template.story[0] and hasattr(doc.page_template.story[0], 'text') and doc.page_template.story[0].text == text:
                    self.page_layout[story_index] = (level, text, canvas.getPageNumber())
                
        # Call the standard page template
        doc.page_template._page_template(canvas, doc)


# --- Main Execution Block ---

def main():
    """The main execution function."""
    LOGGER.info(f"{REPORT_TITLE} - Script Started.")
    
    # 1. Calculate the monthly window
    report_run_time = datetime.now()
    window_start = report_run_time - timedelta(days=REPORT_WINDOW_DAYS)
    LOGGER.info(f"Report Window: {window_start.strftime('%Y-%m-%d')} to {report_run_time.strftime('%Y-%m-%d')}")

    # 2. Database Setup and Connection
    try:
        db_manager = DBManager()
        db_manager.setup_schema()
    except Exception as e:
        LOGGER.critical(f"Database setup failed. Cannot continue. Error: {e}")
        return

    # 3. Data Gathering and Processing
    all_scraped_items: List[Dict[str, Any]] = []
    
    for topic, wiki_title in TOPICS.items():
        try:
            items = process_topic(topic, wiki_title)
            all_scraped_items.extend(items)
        except Exception as e:
            LOGGER.error(f"Failed to process topic '{topic}': {e}", exc_info=True)

    LOGGER.info(f"Total structured items collected: {len(all_scraped_items)}")

    # 4. Database Storage (Idempotency/Upsert)
    try:
        db_manager.upsert_items(all_scraped_items)
    except Exception as e:
        LOGGER.critical(f"Database upsert operation failed. Error: {e}")
        return

    # 5. Fetch Final Report Data (including potentially older data that is relevant)
    try:
        report_data = db_manager.fetch_report_data(window_start)
    except Exception as e:
        LOGGER.critical(f"Failed to fetch report data. Error: {e}")
        return

    # 6. PDF Generation
    try:
        pdf_generator = PDFReportGenerator(PDF_FILE, report_run_time)
        pdf_generator.generate_report(report_data)
    except Exception as e:
        LOGGER.critical(f"PDF Generation failed. Error: {e}")
        return

    LOGGER.info(f"{REPORT_TITLE} - Script Finished. Output: {PDF_FILE}")

if __name__ == "__main__":
    main()

# End of script
