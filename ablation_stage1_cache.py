"""
Ablation Study — Stage 1: 10-K Filing Cache Builder
=====================================================
Run this ONCE to download and store all 10-K filings locally.
Subsequent experiment runs (Stage 3) will read from this cache only —
no SEC requests during the actual ablation loops.

Usage:
    python ablation_stage1_cache.py

Output:
    ablation_cache.db  (SQLite)
        └── filings table: ticker, cik, filing_date, url, raw_text, fetched_at
"""

import os
import re
import time
import sqlite3
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH = "ablation_cache.db"

FILING_RANGE = ("2024-01-01", "2025-05-15")  # Same range as Part 2

IT_TICKERS = [
    'MSFT', 'AAPL', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'CSCO', 'AMD', 'QCOM',
    'TXN', 'INTU', 'IBM', 'AMAT', 'MU', 'NOW', 'LRCX', 'ADI', 'PANW', 'KLAC',
    'SNPS', 'CDNS', 'MSI', 'APH', 'CDW', 'TEL', 'FTNT', 'ANET', 'KEYS', 'GLW',
    'TER', 'STX', 'NTAP', 'FSLR', 'TYL', 'AKAM', 'GEN', 'JNPR', 'QRVO', 'SWKS',
    'WDC', 'ENPH', 'TRMB', 'ZBRA', 'PTC', 'VRT', 'ETN', 'HPE'
]

TICKER_TO_CIK = {
    'MSFT': '0000789019', 'AAPL': '0000320193', 'NVDA': '0001045810', 'AVGO': '0001730168',
    'ORCL': '0001341439', 'CRM': '0001108524', 'ADBE': '0000796343', 'CSCO': '0000858877',
    'AMD': '0000002488', 'QCOM': '0000804328', 'TXN': '0000097476', 'INTU': '0000896878',
    'IBM': '0000051143', 'AMAT': '0000006951', 'MU': '0000723125', 'NOW': '0001373715',
    'LRCX': '0000707549', 'ADI': '0000006281', 'PANW': '0001327567', 'KLAC': '0000319201',
    'SNPS': '0000883241', 'CDNS': '0000813672', 'MSI': '0000068505', 'APH': '0000820313',
    'CDW': '0001402057', 'TEL': '0001385157', 'FTNT': '0001262039', 'ANET': '0001596532',
    'KEYS': '0001601046', 'GLW': '0000024741', 'TER': '0000097210', 'STX': '0001137789',
    'NTAP': '0001002047', 'FSLR': '0001274494', 'TYL': '0000860731', 'AKAM': '0001086222',
    'GEN': '0000849399', 'JNPR': '0001043604', 'QRVO': '0001602658', 'SWKS': '0000004127',
    'WDC': '0000106040', 'ENPH': '0001463101', 'TRMB': '0000864749', 'ZBRA': '0000877212',
    'PTC': '0000857005', 'VRT': '0001674101', 'ETN': '0001551182', 'HPE': '0001645590'
}

TICKER_TO_NAME = {
    'MSFT': 'Microsoft', 'AAPL': 'Apple', 'NVDA': 'NVIDIA', 'AVGO': 'Broadcom',
    'ORCL': 'Oracle', 'CRM': 'Salesforce', 'ADBE': 'Adobe', 'CSCO': 'Cisco Systems',
    'AMD': 'Advanced Micro Devices', 'QCOM': 'QUALCOMM', 'TXN': 'Texas Instruments',
    'INTU': 'Intuit', 'IBM': 'International Business Machines', 'AMAT': 'Applied Materials',
    'MU': 'Micron Technology', 'NOW': 'ServiceNow', 'LRCX': 'Lam Research',
    'ADI': 'Analog Devices', 'PANW': 'Palo Alto Networks', 'KLAC': 'KLA',
    'SNPS': 'Synopsys', 'CDNS': 'Cadence Design Systems', 'MSI': 'Motorola Solutions',
    'APH': 'Amphenol', 'CDW': 'CDW', 'TEL': 'TE Connectivity', 'FTNT': 'Fortinet',
    'ANET': 'Arista Networks', 'KEYS': 'Keysight Technologies', 'GLW': 'Corning',
    'TER': 'Teradyne', 'STX': 'Seagate Technology', 'NTAP': 'NetApp',
    'FSLR': 'First Solar', 'TYL': 'Tyler Technologies', 'AKAM': 'Akamai Technologies',
    'GEN': 'Gen Digital', 'JNPR': 'Juniper Networks', 'QRVO': 'Qorvo',
    'SWKS': 'Skyworks Solutions', 'WDC': 'Western Digital', 'ENPH': 'Enphase Energy',
    'TRMB': 'Trimble', 'ZBRA': 'Zebra Technologies', 'PTC': 'PTC',
    'VRT': 'Vertiv Holdings', 'ETN': 'Eaton', 'HPE': 'Hewlett Packard Enterprise'
}

# Per SEC fair-access policy — replace with your email
SEC_EMAIL = os.environ.get("SEC_EMAIL", "YOUR_EMAIL@domain.com")
HEADERS = {
    'User-Agent': f'Academic ResearchProject {SEC_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}


# ── DB Initialization ─────────────────────────────────────────────────────────

def init_db(db_path: str):
    """Create tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            ticker       TEXT PRIMARY KEY,
            cik          TEXT,
            filing_date  TEXT,
            url          TEXT,
            raw_text     TEXT,
            char_count   INTEGER,
            fetched_at   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snippets (
            ticker_a   TEXT,
            ticker_b   TEXT,
            snippet    TEXT,
            PRIMARY KEY (ticker_a, ticker_b)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_results (
            condition    TEXT,
            model        TEXT,
            prompt_type  TEXT,
            run_id       INTEGER,
            ticker_a     TEXT,
            ticker_b     TEXT,
            rel_type     TEXT,
            confidence   REAL,
            direction    TEXT,
            weight       REAL,
            created_at   TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("DB initialized:", db_path)


# ── SEC Helpers ───────────────────────────────────────────────────────────────

def get_10k_url(cik: str, filing_range: tuple) -> str | None:
    """Fetch the 10-K filing URL from SEC EDGAR within the given date range."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    start, end = filing_range
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            return None
        submissions = res.json()['filings']['recent']
        for i, form in enumerate(submissions['form']):
            filing_date = submissions['filingDate'][i]
            if form == '10-K' and start <= filing_date <= end:
                acc = submissions['accessionNumber'][i].replace('-', '')
                doc = submissions['primaryDocument'][i]
                cik_stripped = cik.lstrip('0')
                return f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc}/{doc}"
    except Exception:
        pass
    return None


def fetch_and_clean_text(url: str) -> str | None:
    """Download 10-K HTML and return cleaned plain text."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=30)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).lower().strip()
        return text
    except Exception:
        return None


# ── Stage 1: Filing Cache ─────────────────────────────────────────────────────

def build_filing_cache(
    tickers: list,
    filing_range: tuple,
    db_path: str = DB_PATH
):
    """
    Download 10-K filings from SEC EDGAR and store in SQLite.
    Already-cached tickers are skipped automatically.

    Expected runtime: ~15 minutes for 48 companies.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)

    # Check what's already cached
    cached = pd.read_sql("SELECT ticker FROM filings", conn)['ticker'].tolist()
    remaining = [t for t in tickers if t not in cached]

    print(f"Already cached : {len(cached)} companies")
    print(f"To fetch       : {len(remaining)} companies")
    print("-" * 50)

    success, failed = 0, []

    for i, ticker in enumerate(remaining, 1):
        cik = TICKER_TO_CIK.get(ticker)
        if not cik:
            print(f"[{i:02d}/{len(remaining)}] {ticker} — CIK not found, skipping")
            failed.append(ticker)
            continue

        print(f"[{i:02d}/{len(remaining)}] {ticker} — fetching...", end=" ")

        # Get filing URL
        url = get_10k_url(cik, filing_range)
        if not url:
            print("❌ No 10-K URL found")
            failed.append(ticker)
            time.sleep(0.2)
            continue

        # Download and clean text
        time.sleep(0.3)  # Respect SEC rate limit
        text = fetch_and_clean_text(url)
        if not text:
            print("❌ Failed to fetch document")
            failed.append(ticker)
            continue

        # Save to DB
        conn.execute("""
            INSERT OR REPLACE INTO filings
            (ticker, cik, filing_date, url, raw_text, char_count, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, cik, filing_range[1], url, text, len(text),
              datetime.now().isoformat()))
        conn.commit()

        print(f"✅ {len(text):,} chars saved")
        success += 1

    conn.close()

    print("\n" + "=" * 50)
    print(f"Stage 1 Complete")
    print(f"  Success : {success}")
    print(f"  Failed  : {len(failed)} {failed if failed else ''}")
    print(f"  DB      : {db_path}")
    print("=" * 50)


# ── Stage 2: Snippet Cache ────────────────────────────────────────────────────

def build_snippet_cache(db_path: str = DB_PATH):
    """
    Pre-extract all company-mention snippets from cached 10-K text.
    Produces one row per (company_A, company_B) pair where B is mentioned in A's filing.

    Total pairs: 48 × 47 = 2,256 max
    Expected runtime: ~2 minutes (pure Python, no network calls).
    """
    conn = sqlite3.connect(db_path)

    filings = pd.read_sql("SELECT ticker, raw_text FROM filings", conn)
    if filings.empty:
        print("❌ No filings found. Run build_filing_cache() first.")
        conn.close()
        return

    print(f"Building snippets from {len(filings)} cached filings...")

    snippets = []
    total_pairs = len(filings) * (len(filings) - 1)

    for _, row_a in filings.iterrows():
        ticker_a = row_a['ticker']
        text_a   = row_a['raw_text']

        for _, row_b in filings.iterrows():
            ticker_b = row_b['ticker']
            if ticker_a == ticker_b:
                continue

            name_b = TICKER_TO_NAME.get(ticker_b, '').lower()
            if not name_b or name_b not in text_a:
                continue

            idx     = text_a.find(name_b)
            snippet = text_a[max(0, idx - 150): idx + 150]

            snippets.append({
                'ticker_a': ticker_a,
                'ticker_b': ticker_b,
                'snippet' : snippet
            })

    # Save to DB (replace if re-running)
    if snippets:
        conn.execute("DELETE FROM snippets")
        pd.DataFrame(snippets).to_sql(
            'snippets', conn, if_exists='append', index=False
        )
        conn.commit()

    conn.close()

    print("=" * 50)
    print(f"Stage 2 Complete")
    print(f"  Total pairs scanned : {total_pairs:,}")
    print(f"  Snippets extracted  : {len(snippets):,}")
    print(f"  Coverage            : {len(snippets)/total_pairs*100:.1f}%")
    print("=" * 50)


# ── Cache Inspection ──────────────────────────────────────────────────────────

def inspect_cache(db_path: str = DB_PATH):
    """Print a summary of what's currently in the cache."""
    conn = sqlite3.connect(db_path)

    filings  = pd.read_sql("SELECT ticker, char_count, fetched_at FROM filings", conn)
    snippets = pd.read_sql("SELECT COUNT(*) as cnt FROM snippets", conn)

    print("\n── Cached Filings ──────────────────────────────")
    if filings.empty:
        print("  None yet.")
    else:
        print(filings.to_string(index=False))
        print(f"\n  Total: {len(filings)} companies")
        print(f"  Avg size: {filings['char_count'].mean():,.0f} chars")

    print(f"\n── Snippets ────────────────────────────────────")
    print(f"  Total: {snippets['cnt'].iloc[0]:,} pairs")

    conn.close()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Ablation Study — Stage 1 & 2: Building Cache")
    print("=" * 50)

    # Stage 1: Download 10-K filings
    build_filing_cache(IT_TICKERS, FILING_RANGE)

    # Stage 2: Extract snippets
    build_snippet_cache()

    # Inspect result
    inspect_cache()