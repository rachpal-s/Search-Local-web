"""scraper/cleaner.py — Table extraction and noise removal utilities."""
import re
from typing import Optional
from bs4 import BeautifulSoup, Tag


NOISE_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", "noscript",
    "form", "iframe", "svg", "button", "input", "select", "textarea",
]

AD_PAT = re.compile(
    r"(^ad[-_]|advertisement|sponsor|promo|banner|cookie|popup|modal|"
    r"overlay|sidebar|taboola|outbrain|dfp|gpt-ad|subscribe|newsletter)",
    re.I,
)


def extract_tables_markdown(html: str) -> tuple[str, int]:
    """
    Extract all meaningful data tables from HTML as Markdown.
    Returns (markdown_string, table_count).
    Skips layout tables (too few columns, no thead, all same-length cells).
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise first
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    tables = soup.find_all("table")
    out = []
    table_count = 0

    for table in tables:
        if not isinstance(table, Tag):
            continue

        # ── Get headers ───────────────────────────────────────────────────────
        headers: list[str] = []
        thead = table.find("thead")
        if thead and isinstance(thead, Tag):
            header_cells = thead.find_all(["th", "td"])
            headers = [_cell_text(c) for c in header_cells]
            headers = [h for h in headers if h]

        # If no thead, check first <tr> for <th> elements
        if not headers:
            first_tr = table.find("tr")
            if first_tr and isinstance(first_tr, Tag):
                ths = first_tr.find_all("th")
                if ths:
                    headers = [_cell_text(th) for th in ths]

        # ── Get rows ──────────────────────────────────────────────────────────
        tbody = table.find("tbody") or table
        rows: list[list[str]] = []
        for tr in tbody.find_all("tr"):
            if not isinstance(tr, Tag):
                continue
            cells = [_cell_text(td) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells and not all(c == cells[0] for c in cells):  # skip uniform rows
                rows.append(cells)

        # ── Skip degenerate tables ────────────────────────────────────────────
        if not rows:
            continue
        max_cols = max(len(r) for r in rows)
        if max_cols < 2:
            continue
        if len(rows) < 2 and not headers:
            continue

        table_count += 1

        # ── Infer title from surrounding context ──────────────────────────────
        caption = table.find("caption")
        section_title = None
        for sib in table.find_all_previous(["h1","h2","h3","h4","h5"], limit=1):
            section_title = sib.get_text(strip=True)
            break

        if caption:
            out.append(f"### {caption.get_text(strip=True)}")
        elif section_title:
            out.append(f"### {section_title}")

        # ── Normalise column count ────────────────────────────────────────────
        if not headers:
            headers = [f"Col {i+1}" for i in range(max_cols)]

        n_cols = len(headers)
        out.append("| " + " | ".join(headers) + " |")
        out.append("| " + " | ".join(["---"] * n_cols) + " |")

        for row in rows:
            padded = row[:n_cols] + [""] * max(0, n_cols - len(row))
            out.append("| " + " | ".join(padded) + " |")

        out.append("")

    return "\n".join(out).strip(), table_count


def _cell_text(tag: Tag) -> str:
    """Get clean text from a table cell, collapsing whitespace."""
    if not isinstance(tag, Tag):
        return ""
    text = tag.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def remove_noise(html: str) -> str:
    """Strip scripts, ads, navs etc. and return cleaned HTML."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for el in list(soup.find_all(True)):
        if not isinstance(el, Tag) or el.parent is None:
            continue
        cls = " ".join(el.get("class") or [])
        eid = el.get("id") or ""
        if AD_PAT.search(cls) or AD_PAT.search(eid):
            el.decompose()
    return str(soup)
