"""Patent Reader API — fetch and return structured patent data from Google Patents."""

import logging
import re
from dataclasses import dataclass, field, asdict

from collections import Counter, defaultdict

import httpx
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patents", tags=["patents"])


# ── Structured types ──────────────────────────────────────────────────────


@dataclass
class PatentClaim:
    number: int
    text: str
    depends_on: int | None  # None = independent
    type: str  # "independent" | "dependent"


@dataclass
class PatentSection:
    heading: str
    paragraphs: list[str]


@dataclass
class PatentData:
    title: str
    patent_number: str
    filing_date: str
    publication_date: str
    inventors: list[str]
    assignee: str
    classifications: list[str]
    abstract: str
    claims: list[PatentClaim]
    description: list[PatentSection]
    pdf_url: str = ""
    figure_urls: list[str] = field(default_factory=list)


# ── Parsing ───────────────────────────────────────────────────────────────


def _normalize_pub_number(raw: str) -> list[str]:
    """Normalize a publication number into Google Patents URL candidates."""
    cleaned = raw.strip().upper().replace(",", "")
    m = re.match(
        r"([A-Z]{2})\s*"
        r"(\d[\d\s/]*\d)"
        r"\s*([A-Z]\d?)?\s*$",
        cleaned,
    )
    if not m:
        return [re.sub(r"[^A-Z0-9]", "", cleaned)]

    country = m.group(1)
    number = re.sub(r"[\s/]", "", m.group(2))
    kind = m.group(3)
    base = f"{country}{number}"

    if kind:
        return [f"{base}{kind}"]
    return [f"{base}B2", f"{base}B1", f"{base}A1", f"{base}A"]


def _parse_patent_html(html: str, pub_number: str) -> PatentData:
    """Parse Google Patents HTML into structured PatentData."""
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_meta = soup.find("meta", {"name": "DC.title"})
    title = title_meta.get("content", "").strip() if title_meta else pub_number

    # PDF URL
    pdf_meta = soup.find("meta", {"name": "citation_pdf_url"})
    pdf_url = pdf_meta.get("content", "").strip() if pdf_meta else ""

    # Dates — take the first filingDate/publicationDate <time> elements
    filing_date = ""
    pub_date = ""
    el = soup.find("time", itemprop="filingDate")
    if el:
        filing_date = el.get("datetime", "")
    el = soup.find("time", itemprop="publicationDate")
    if el:
        pub_date = el.get("datetime", "")

    # Inventors — only from the first article.result (the patent itself)
    inventors: list[str] = []
    first_article = soup.find("article", class_="result")
    if first_article:
        for inv_el in first_article.find_all(itemprop="inventor"):
            name = inv_el.get_text(strip=True)
            if name:
                inventors.append(name)

    # Assignee — first assigneeOriginal in the first article
    assignee = ""
    if first_article:
        a_el = first_article.find(itemprop="assigneeOriginal")
        if a_el:
            assignee = a_el.get_text(strip=True)

    # Classifications — CPC codes from itemprop="Code" under itemprop="classifications"
    classifications: list[str] = []
    for code_el in soup.find_all("span", itemprop="Code"):
        code = code_el.get_text(strip=True)
        parent = code_el.parent
        if parent and parent.get("itemprop") == "classifications" and len(code) > 3:
            # Take only the most specific codes (contain /)
            if "/" in code:
                classifications.append(code)
    # Deduplicate while preserving order, limit to first 8
    classifications = list(dict.fromkeys(classifications))[:8]

    # Abstract
    abstract = ""
    abstract_section = soup.find("section", itemprop="abstract")
    if abstract_section:
        content_div = abstract_section.find("div", itemprop="content")
        if content_div:
            abstract = content_div.get_text(separator=" ", strip=True)

    # Description — extract headings and paragraph groups
    # Google Patents uses either <ul class="description"> or <div class="description">
    sections: list[PatentSection] = []
    desc_section = soup.find("section", itemprop="description")
    if desc_section:
        content_div = desc_section.find("div", itemprop="content")
        if content_div:
            desc_container = (
                content_div.find("ul", class_="description")
                or content_div.find("div", class_="description")
            )
            if desc_container:
                current_heading = "DESCRIPTION"
                current_paras: list[str] = []
                for child in desc_container.children:
                    if not isinstance(child, Tag):
                        continue
                    if child.name == "heading":
                        if current_paras:
                            sections.append(PatentSection(heading=current_heading, paragraphs=current_paras))
                            current_paras = []
                        current_heading = child.get_text(strip=True)
                    elif child.name in ("li", "p", "div"):
                        # Skip nested description containers
                        if "description" in (child.get("class") or []):
                            continue
                        text = child.get_text(separator=" ", strip=True)
                        if text:
                            current_paras.append(text)
                if current_paras:
                    sections.append(PatentSection(heading=current_heading, paragraphs=current_paras))

    # Claims — parse with dependency info
    claims: list[PatentClaim] = []
    claims_section = soup.find("section", itemprop="claims")
    if claims_section:
        claims_div = claims_section.find("div", class_="claims")
        if claims_div:
            for claim_div in claims_div.find_all("div", class_="claim", attrs={"num": True}):
                num_str = claim_div["num"].lstrip("0") or "0"
                try:
                    num = int(num_str)
                except ValueError:
                    continue

                text = claim_div.get_text(separator=" ", strip=True)

                # Extract dependency from claim-ref elements
                depends_on: int | None = None
                claim_refs = claim_div.find_all("claim-ref")
                for ref in claim_refs:
                    idref = ref.get("idref", "")
                    m = re.search(r"(\d+)", idref)
                    if m:
                        depends_on = int(m.group(1))
                        break

                claims.append(PatentClaim(
                    number=num,
                    text=text,
                    depends_on=depends_on,
                    type="dependent" if depends_on is not None else "independent",
                ))

    # Figures — extract full-resolution image URLs from <meta itemprop="full"> elements
    # (the sibling <img itemprop="thumbnail"> contains low-res 111px thumbnails)
    figure_urls: list[str] = []
    for meta in soup.find_all("meta", itemprop="full"):
        url = meta.get("content", "")
        if "patentimages.storage.googleapis.com" in url and url.endswith(".png"):
            figure_urls.append(url)

    return PatentData(
        title=title,
        patent_number=pub_number,
        filing_date=filing_date,
        publication_date=pub_date,
        inventors=inventors,
        assignee=assignee,
        classifications=classifications,
        abstract=abstract,
        claims=claims,
        description=sections,
        pdf_url=pdf_url,
        figure_urls=figure_urls,
    )


# ── Endpoint ──────────────────────────────────────────────────────────────


@router.get("/{publication_number}")
async def get_patent(publication_number: str):
    """Fetch and return structured patent data from Google Patents.

    Accepts formats like US11423567B2, US-11423567-B2, US 11,423,567 B2, etc.
    """
    candidates = _normalize_pub_number(publication_number)

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        for candidate in candidates:
            url = f"https://patents.google.com/patent/{candidate}/en"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = _parse_patent_html(resp.text, candidate)
                    return asdict(data)
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", candidate, exc)
                continue

    raise HTTPException(
        status_code=404,
        detail=f"Patent not found. Tried: {', '.join(candidates)}",
    )


# ── Reference numerals ───────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the of to for with by from on in and or said each at least one "
    "its is are that this be wherein further".split()
)

# Matches "some words ( 110 )" or "some words ( 110a )"
_REF_PATTERN = re.compile(
    r"((?:\b[\w-]+\s+){1,5})\(\s*(\d+[a-zA-Z]?)\s*\)"
)


def _extract_reference_numerals(data: PatentData) -> list[dict]:
    """Extract reference numeral → label mappings from patent text.

    Scans abstract, description, and claims for patterns like "camera ( 110 )"
    and returns a deduplicated list sorted by numeral.
    """
    # Collect all text
    parts = [data.abstract]
    for section in data.description:
        parts.extend(section.paragraphs)
    for claim in data.claims:
        parts.append(claim.text)
    all_text = " ".join(parts)

    # Find all (context, numeral) pairs
    num_labels: defaultdict[str, list[str]] = defaultdict(list)
    for match in _REF_PATTERN.finditer(all_text):
        context = match.group(1).strip()
        numeral = match.group(2)

        # Clean label: take last 1-4 words, strip leading/trailing stop words
        words = context.split()[-4:]
        while words and words[0].lower() in _STOP_WORDS:
            words = words[1:]
        while words and words[-1].lower() in _STOP_WORDS:
            words = words[:-1]
        if not words:
            continue
        label = " ".join(words).lower()
        # Strip common prepositional noise from label
        label = re.sub(r"^.*?\b(?:of|from|on|via|into|over)\s+(?:a|an|the)\s+", "", label)
        if not label:
            continue

        # Skip likely method step numbers (single word that's a verb/gerund)
        if len(words) == 1 and words[0].endswith("ing"):
            continue

        num_labels[numeral].append(label)

    # Pick best label: among labels seen >1 time, prefer shortest; fallback to overall most common
    results = []
    for numeral, labels in num_labels.items():
        counts = Counter(labels)
        # Labels seen more than once
        repeated = {l: c for l, c in counts.items() if c > 1}
        if repeated:
            best = min(repeated, key=len)
        else:
            best = counts.most_common(1)[0][0]
        results.append({
            "numeral": numeral,
            "label": best,
            "count": len(labels),
        })

    # Sort by numeric value
    results.sort(key=lambda r: (int(re.match(r"\d+", r["numeral"]).group()), r["numeral"]))  # type: ignore[union-attr]
    return results


@router.get("/{publication_number}/reference-numerals")
async def get_reference_numerals(publication_number: str):
    """Extract reference numeral → label mappings from a patent.

    Returns a list of {numeral, label, count} objects sorted by numeral.
    """
    candidates = _normalize_pub_number(publication_number)

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        for candidate in candidates:
            url = f"https://patents.google.com/patent/{candidate}/en"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = _parse_patent_html(resp.text, candidate)
                    return {"numerals": _extract_reference_numerals(data)}
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", candidate, exc)
                continue

    raise HTTPException(
        status_code=404,
        detail=f"Patent not found. Tried: {', '.join(candidates)}",
    )
