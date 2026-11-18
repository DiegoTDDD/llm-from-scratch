"""Download 10-K sections from SEC EDGAR and write the training corpus.

The corpus is the Business, Risk Factors and MD&A sections of the latest two
10-K filings of 30 large US companies across sectors, the same source the
sec-filings-rag project indexes. Whitespace is normalised; nothing else is
changed. Output: data/corpus.txt plus data/corpus_manifest.json listing every
section and its size, so the corpus can be rebuilt and audited.

SEC asks automated clients to identify themselves with a name and a contact
address: set SEC_IDENTITY, e.g. "Jane Doe jane@example.com".

Usage:
    python scripts/prepare_corpus.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from edgar import Company, set_identity

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus.txt"
MANIFEST = ROOT / "data" / "corpus_manifest.json"
DEFAULT_IDENTITY = "llm-from-scratch DiegoTDDD@users.noreply.github.com"
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "WFC",
    "GS", "V", "MA", "JNJ", "PFE", "MRK", "UNH", "XOM", "CVX", "KO",
    "PEP", "WMT", "COST", "HD", "DIS", "NFLX", "INTC", "CSCO", "BA", "CAT",
]
FILINGS_PER_COMPANY = 2
SECTIONS = {
    "business": lambda tenk: tenk.business,
    "risk_factors": lambda tenk: tenk.risk_factors,
    "mda": lambda tenk: tenk.management_discussion,
}
MIN_SECTION_CHARS = 2_000


def clean(text: str) -> str:
    """Collapse runs of spaces and blank lines; keep paragraph breaks."""
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_identity(os.environ.get("SEC_IDENTITY", DEFAULT_IDENTITY))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest, parts = [], []
    for ticker in TICKERS:
        try:
            filings = Company(ticker).get_filings(form="10-K").head(FILINGS_PER_COMPANY)
        except Exception as exc:
            log.warning("%s: no filings (%s)", ticker, exc)
            continue
        for filing in filings:
            try:
                tenk = filing.obj()
            except Exception as exc:
                log.warning("%s %s: could not parse (%s)", ticker, filing.filing_date, exc)
                continue
            for name, getter in SECTIONS.items():
                try:
                    text = clean(getter(tenk) or "")
                except Exception:
                    text = ""
                if len(text) < MIN_SECTION_CHARS:
                    continue
                parts.append(text)
                manifest.append({"ticker": ticker, "filing_date": str(filing.filing_date),
                                 "section": name, "chars": len(text)})
            log.info("%s %s: %d sections so far", ticker, filing.filing_date, len(manifest))
            time.sleep(0.2)  # stay well under SEC's 10 requests per second
    corpus = "\n\n".join(parts)
    OUT.write_text(corpus, encoding="utf-8")
    MANIFEST.write_text(json.dumps({"sections": manifest, "total_chars": len(corpus)}, indent=1),
                        encoding="utf-8")
    log.info("Wrote %s characters from %d sections to %s", f"{len(corpus):,}", len(manifest), OUT)


if __name__ == "__main__":
    main()
