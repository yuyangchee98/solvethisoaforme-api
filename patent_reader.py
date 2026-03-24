"""Patent Reader API — fetch and return structured patent data from Google Patents."""

import asyncio
import base64
import logging
import os
import re
import struct
from dataclasses import dataclass, field, asdict

from collections import Counter, defaultdict

import httpx
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException

from core.nlp_models import nlp as spacy_nlp

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
class PatentParagraph:
    text: str
    number: str | None = None  # e.g. "0001", absent for some patents


@dataclass
class PatentSection:
    heading: str
    paragraphs: list[PatentParagraph]


@dataclass
class PatentCitation:
    publication_number: str
    priority_date: str
    publication_date: str
    assignee: str
    title: str
    examiner_cited: bool


@dataclass
class NonPatentCitation:
    title: str


@dataclass
class FamilyApplication:
    application_number: str
    representative_publication: str
    priority_date: str
    filing_date: str
    title: str
    status: str
    expiration: str


@dataclass
class CountryStatus:
    country_code: str
    publication_number: str
    count: int


@dataclass
class LegalEvent:
    date: str
    code: str
    title: str
    attributes: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SimilarDocument:
    publication_number: str
    publication_date: str
    title: str


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
    priority_date: str = ""
    patent_citations: list[PatentCitation] = field(default_factory=list)
    cited_by: list[PatentCitation] = field(default_factory=list)
    non_patent_citations: list[NonPatentCitation] = field(default_factory=list)
    family_applications: list[FamilyApplication] = field(default_factory=list)
    country_status: list[CountryStatus] = field(default_factory=list)
    legal_events: list[LegalEvent] = field(default_factory=list)
    similar_documents: list[SimilarDocument] = field(default_factory=list)


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


def _parse_citation_rows(soup: BeautifulSoup, itemprop: str) -> list[PatentCitation]:
    """Parse patent citation rows (backwardReferences or forwardReferencesOrig)."""
    citations = []
    for row in soup.find_all("tr", itemprop=itemprop):
        pub_el = row.find("span", itemprop="publicationNumber")
        pub_number = pub_el.get_text(strip=True) if pub_el else ""
        if not pub_number:
            continue

        priority = ""
        td = row.find("td", itemprop="priorityDate")
        if td:
            priority = td.get_text(strip=True)

        pub_date = ""
        td = row.find("td", itemprop="publicationDate")
        if td:
            pub_date = td.get_text(strip=True)

        assignee_el = row.find("span", itemprop="assigneeOriginal")
        assignee = assignee_el.get_text(strip=True) if assignee_el else ""

        title_td = row.find("td", itemprop="title")
        title = title_td.get_text(strip=True) if title_td else ""

        examiner_el = row.find("span", itemprop="examinerCited")
        examiner_cited = examiner_el is not None and examiner_el.get_text(strip=True) == "*"

        citations.append(PatentCitation(
            publication_number=pub_number,
            priority_date=priority,
            publication_date=pub_date,
            assignee=assignee,
            title=title,
            examiner_cited=examiner_cited,
        ))
    return citations


def _parse_non_patent_citations(soup: BeautifulSoup) -> list[NonPatentCitation]:
    """Parse non-patent literature citation rows."""
    citations = []
    for row in soup.find_all("tr", itemprop="detailedNonPatentLiterature"):
        title_el = row.find("span", itemprop="title")
        if title_el:
            text = title_el.get_text(separator=" ", strip=True)
            if text:
                citations.append(NonPatentCitation(title=text))
    return citations


def _parse_family_applications(soup: BeautifulSoup) -> list[FamilyApplication]:
    """Parse patent family application rows."""
    apps = []
    for row in soup.find_all("tr", itemprop="applications"):
        app_el = row.find("span", itemprop="applicationNumber")
        app_number = app_el.get_text(strip=True) if app_el else ""

        rep_el = row.find("span", itemprop="representativePublication")
        rep_pub = rep_el.get_text(strip=True) if rep_el else ""

        priority = ""
        td = row.find("td", itemprop="priorityDate")
        if td:
            priority = td.get_text(strip=True)

        filing = ""
        td = row.find("td", itemprop="filingDate")
        if td:
            filing = td.get_text(strip=True)

        title_td = row.find("td", itemprop="title")
        title = title_td.get_text(strip=True) if title_td else ""

        status_el = row.find("span", itemprop="ifiStatus")
        status = status_el.get_text(strip=True) if status_el else ""

        exp_el = row.find("span", itemprop="ifiExpiration")
        expiration = exp_el.get_text(strip=True) if exp_el else ""

        if app_number or rep_pub:
            apps.append(FamilyApplication(
                application_number=app_number,
                representative_publication=rep_pub,
                priority_date=priority,
                filing_date=filing,
                title=title,
                status=status,
                expiration=expiration,
            ))
    return apps


def _parse_country_status(soup: BeautifulSoup) -> list[CountryStatus]:
    """Parse patent family country status rows."""
    statuses = []
    for row in soup.find_all("tr", itemprop="countryStatus"):
        code_el = row.find("span", itemprop="countryCode")
        code = code_el.get_text(strip=True) if code_el else ""

        num_el = row.find("span", itemprop="num")
        count = 1
        if num_el:
            try:
                count = int(num_el.get_text(strip=True))
            except ValueError:
                pass

        pub_el = row.find("span", itemprop="representativePublication")
        pub = pub_el.get_text(strip=True) if pub_el else ""

        if code:
            statuses.append(CountryStatus(
                country_code=code,
                publication_number=pub,
                count=count,
            ))
    return statuses


def _parse_legal_events(soup: BeautifulSoup) -> list[LegalEvent]:
    """Parse patent legal event rows."""
    events = []
    for row in soup.find_all("tr", itemprop="legalEvents"):
        time_el = row.find("time", itemprop="date")
        date = time_el.get("datetime", "") if time_el else ""

        code_td = row.find("td", itemprop="code")
        code = code_td.get_text(strip=True) if code_td else ""

        title_td = row.find("td", itemprop="title")
        title = title_td.get_text(strip=True) if title_td else ""

        attributes = []
        for attr_p in row.find_all("p", itemprop="attributes"):
            label_el = attr_p.find("strong", itemprop="label")
            value_el = attr_p.find("span", itemprop="value")
            label = label_el.get_text(strip=True) if label_el else ""
            value = value_el.get_text(strip=True) if value_el else ""
            if label or value:
                attributes.append({"label": label, "value": value})

        if date or code:
            events.append(LegalEvent(
                date=date,
                code=code,
                title=title,
                attributes=attributes,
            ))
    return events


def _parse_similar_documents(soup: BeautifulSoup, self_number: str) -> list[SimilarDocument]:
    """Parse similar documents rows, excluding the patent itself."""
    self_normalized = re.sub(r"[^A-Z0-9]", "", self_number.upper())
    docs = []
    for row in soup.find_all("tr", itemprop="similarDocuments"):
        pub_el = row.find("span", itemprop="publicationNumber")
        pub_number = pub_el.get_text(strip=True) if pub_el else ""

        if not pub_number or re.sub(r"[^A-Z0-9]", "", pub_number.upper()) == self_normalized:
            continue

        time_el = row.find("time", itemprop="publicationDate")
        pub_date = time_el.get("datetime", "") if time_el else ""

        title_td = row.find("td", itemprop="title")
        title = title_td.get_text(strip=True) if title_td else ""

        docs.append(SimilarDocument(
            publication_number=pub_number,
            publication_date=pub_date,
            title=title,
        ))
    return docs


def _parse_patent_html(html: str, pub_number: str) -> PatentData:
    """Parse Google Patents HTML into structured PatentData."""
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_meta = soup.find("meta", {"name": "DC.title"})
    title = title_meta.get("content", "").strip() if title_meta else pub_number

    # PDF URL
    pdf_meta = soup.find("meta", {"name": "citation_pdf_url"})
    pdf_url = pdf_meta.get("content", "").strip() if pdf_meta else ""

    # Dates — take the first filingDate/publicationDate/priorityDate <time> elements
    filing_date = ""
    pub_date = ""
    priority_date = ""
    el = soup.find("time", itemprop="filingDate")
    if el:
        filing_date = el.get("datetime", "")
    el = soup.find("time", itemprop="publicationDate")
    if el:
        pub_date = el.get("datetime", "")
    el = soup.find("time", itemprop="priorityDate")
    if el:
        priority_date = el.get("datetime", "")

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
                current_paras: list[PatentParagraph] = []
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
                            num = child.get("num") or None
                            current_paras.append(PatentParagraph(text=text, number=num))
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
                # Strip leading claim number — Google Patents embeds it as
                # "1." (granted B1/B2) or "1 ." (applications A1)
                text = re.sub(r"^\d+\s*\.\s*", "", text)

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
        priority_date=priority_date,
        patent_citations=_parse_citation_rows(soup, "backwardReferences"),
        cited_by=_parse_citation_rows(soup, "forwardReferencesOrig"),
        non_patent_citations=_parse_non_patent_citations(soup),
        family_applications=_parse_family_applications(soup),
        country_status=_parse_country_status(soup),
        legal_events=_parse_legal_events(soup),
        similar_documents=_parse_similar_documents(soup, pub_number),
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

_REF_NUM_PAT = re.compile(r"^\d{1,5}[a-zA-Z]?$")

# Labels to skip — document structure references, not element names
_SKIP_LABELS = frozenset(
    "fig figs figure figures claim claims step steps page pages "
    "table tables example examples section sections paragraph paragraphs "
    "chapter chapters col row eq equ embodiment embodiments".split()
)

_FIG_ENUM = re.compile(
    r"(?:figs?\.?|figures?)\s+"           # trigger word
    r"\d+[a-zA-Z]?"                       # first number
    r"(?:"
    r"  \s*,\s*\d+[a-zA-Z]?"             # comma-separated continuations
    r"| \s+(?:and|or|to|through|-)\s+\d+[a-zA-Z]?"  # conjunction continuations
    r")+",
    re.I | re.X,
)
_FIG_CONTINUATION_NUM = re.compile(r"\d+[a-zA-Z]?")


def _build_fig_exclusion_zones(text: str) -> list[tuple[int, int]]:
    """Find continuation numbers in figure enumerations like 'FIGS. 2 and 3'.

    Returns character spans for every number AFTER the first in each enumeration.
    """
    zones = []
    for m in _FIG_ENUM.finditer(text):
        nums = list(_FIG_CONTINUATION_NUM.finditer(text, m.start(), m.end()))
        for num_match in nums[1:]:  # skip first number
            zones.append((num_match.start(), num_match.end()))
    return zones


def _build_exclusion_zones(text: str):
    """Run spaCy NER and return (doc, exclusion_zones).

    Exclusion zones are character spans for dates, quantities, units, and
    figure enumeration continuations — positions where a number is NOT a
    reference numeral.
    """
    doc = spacy_nlp(text)
    _EXCLUDE_LABELS = {"DATE", "QUANTITY", "TIME", "PERCENT"}
    zones = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in _EXCLUDE_LABELS]
    zones.extend(_build_fig_exclusion_zones(text))
    zones.sort()
    return doc, zones


def _in_exclusion_zone(pos: int, zones: list[tuple[int, int]]) -> bool:
    """Check if a character position falls inside any exclusion zone (binary search)."""
    import bisect
    idx = bisect.bisect_right(zones, (pos, float("inf"))) - 1
    if idx >= 0:
        start, end = zones[idx]
        if start <= pos < end:
            return True
    return False


def _chunk_label(chunk) -> str | None:
    """Extract a clean label from a spaCy noun chunk.

    Strips determiners, pronouns, adpositions, conjunctions, auxiliaries,
    adverbs, particles, punctuation, and verbs. Returns None if nothing
    meaningful remains.
    """
    _STRIP_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "ADV", "PART", "PUNCT"}
    tokens = list(chunk)
    # Strip junk POS from the left — but keep past participles (VBN) acting as adjectives
    while tokens and (tokens[0].pos_ in _STRIP_POS
                       or (tokens[0].pos_ == "VERB" and tokens[0].tag_ != "VBN")):
        tokens = tokens[1:]
    # Strip from both ends: punctuation chars (regardless of POS tag)
    while tokens and not tokens[0].text.isalnum():
        tokens = tokens[1:]
    while tokens and not tokens[-1].text.isalnum():
        tokens = tokens[:-1]
    if not tokens:
        return None
    label = " ".join(t.text.lower() for t in tokens)
    # Clean up possessive tokenization: "subject 's eye" → "subject's eye"
    label = label.replace(" 's", "'s")
    # Skip labels that are document structure references
    clean = re.sub(r"[.\s]+$", "", label)
    if clean in _SKIP_LABELS:
        return None
    return label


def _extract_reference_numerals(data: PatentData) -> dict:
    """Extract reference numeral → label mappings and highlight positions from patent text.

    Uses spaCy noun chunks to find element labels adjacent to reference numerals,
    and NER-based exclusion zones to filter out dates, quantities, units, etc.

    Returns {numerals: [...], highlights: {abstract: [...], description: [[...]], claims: [[...]]}}.
    """
    # Build text parts with offset tracking.
    # Each entry: (text, seg_key) where seg_key is ("abstract",) | ("description", si, pi) | ("claims", ci)
    segments: list[tuple[str, tuple]] = []
    segments.append((data.abstract, ("abstract",)))
    for si, section in enumerate(data.description):
        for pi, para in enumerate(section.paragraphs):
            segments.append((para.text, ("description", si, pi)))
    for ci, claim in enumerate(data.claims):
        segments.append((claim.text, ("claims", ci)))

    # Join with spaces and track each segment's start offset in all_text
    seg_offsets: list[tuple[int, int, tuple]] = []  # (start, end, seg_key)
    parts = []
    offset = 0
    for text, key in segments:
        seg_offsets.append((offset, offset + len(text), key))
        parts.append(text)
        offset += len(text) + 1  # +1 for the joining space
    all_text = " ".join(parts)

    doc, exclusion_zones = _build_exclusion_zones(all_text)

    num_labels: defaultdict[str, list[str]] = defaultdict(list)
    # Collect highlight positions as (char_start, char_end, numeral) in all_text
    raw_highlights: list[tuple[int, int, str]] = []

    def _record_match(num_tok, chunk):
        label = _chunk_label(chunk)
        if label:
            num_labels[num_tok.text].append(label)
            raw_highlights.append((num_tok.idx, num_tok.idx + len(num_tok.text), num_tok.text))

    for chunk in doc.noun_chunks:
        end_idx = chunk.end
        if end_idx >= len(doc):
            continue

        next_tok = doc[end_idx]

        # Case 1: bare reference — "camera 110"
        if _REF_NUM_PAT.match(next_tok.text):
            if _in_exclusion_zone(next_tok.idx, exclusion_zones):
                continue
            _record_match(next_tok, chunk)

        # Case 2: parenthesized reference — "camera (110)" / "camera ( 110 )"
        elif next_tok.text == "(":
            paren_idx = end_idx + 1
            while paren_idx < len(doc) and doc[paren_idx].text.isspace():
                paren_idx += 1
            if paren_idx < len(doc) and _REF_NUM_PAT.match(doc[paren_idx].text):
                num_tok = doc[paren_idx]
                if _in_exclusion_zone(num_tok.idx, exclusion_zones):
                    continue
                _record_match(num_tok, chunk)

    # Pick best label
    numerals = []
    for numeral, labels in num_labels.items():
        counts = Counter(labels)
        repeated = {l: c for l, c in counts.items() if c > 1}
        if repeated:
            best = min(repeated, key=len)
        else:
            best = counts.most_common(1)[0][0]
        numerals.append({
            "numeral": numeral,
            "label": best,
            "count": len(labels),
        })
    numerals.sort(key=lambda r: (int(re.match(r"\d+", r["numeral"]).group()), r["numeral"]))  # type: ignore[union-attr]

    # Map highlight positions back to source segments
    import bisect
    seg_starts = [s for s, _, _ in seg_offsets]

    highlights: dict = {
        "abstract": [],
        "description": [[[] for _ in sec.paragraphs] for sec in data.description],
        "claims": [[] for _ in data.claims],
    }
    for abs_start, abs_end, numeral in raw_highlights:
        idx = bisect.bisect_right(seg_starts, abs_start) - 1
        if idx < 0:
            continue
        seg_start, seg_end, seg_key = seg_offsets[idx]
        if abs_start >= seg_end:
            continue  # falls in the joining space
        rel_start = abs_start - seg_start
        rel_end = abs_end - seg_start
        span = {"start": rel_start, "end": rel_end, "numeral": numeral}
        if seg_key[0] == "abstract":
            highlights["abstract"].append(span)
        elif seg_key[0] == "description":
            highlights["description"][seg_key[1]][seg_key[2]].append(span)
        elif seg_key[0] == "claims":
            highlights["claims"][seg_key[1]].append(span)

    return {"numerals": numerals, "highlights": highlights}


@router.get("/{publication_number}/reference-numerals")
async def get_reference_numerals(publication_number: str):
    """Extract reference numeral → label mappings and highlight positions from a patent.

    Returns {numerals: [...], highlights: {abstract: [...], description: [[...]], claims: [[...]]}}.
    """
    candidates = _normalize_pub_number(publication_number)

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        for candidate in candidates:
            url = f"https://patents.google.com/patent/{candidate}/en"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = _parse_patent_html(resp.text, candidate)
                    return _extract_reference_numerals(data)
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", candidate, exc)
                continue

    raise HTTPException(
        status_code=404,
        detail=f"Patent not found. Tried: {', '.join(candidates)}",
    )


# ── Figure map via Google Vision OCR ──────────────────────────────────────

_FIG_PATTERN = re.compile(r"Fig(?:ure)?\s*[.,;:]?\s*(\d+(?:[a-zA-Z]|-\d+)?)", re.IGNORECASE)
_NUMERAL_PATTERN = re.compile(r"^\d{1,5}[a-zA-Z]?$")

_VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Read width and height from a PNG file header."""
    w, h = struct.unpack(">II", data[16:24])
    return w, h


async def _vision_ocr_sheet(
    img_bytes: bytes, client: httpx.AsyncClient,
) -> tuple[list[str], list[dict]]:
    """Send image to Google Vision TEXT_DETECTION.

    Returns (figure_numbers, numeral_bboxes) where numeral_bboxes contains
    normalized 0-1 bounding boxes for every standalone number detected.
    """
    api_key = os.environ.get("GOOGLE_VISION_API_KEY", "")
    if not api_key:
        logger.warning("GOOGLE_VISION_API_KEY not set, skipping figure OCR")
        return [], []

    resp = await client.post(
        _VISION_API_URL,
        params={"key": api_key},
        json={
            "requests": [{
                "image": {"content": base64.b64encode(img_bytes).decode()},
                "features": [{"type": "TEXT_DETECTION"}],
            }]
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        logger.warning("Vision API error %d: %s", resp.status_code, resp.text[:200])
        return [], []

    annotations = resp.json().get("responses", [{}])[0].get("textAnnotations", [])
    if not annotations:
        return [], []

    # Figure numbers from full text block
    full_text = annotations[0]["description"]
    fig_matches = _FIG_PATTERN.findall(full_text)
    figure_numbers = list(dict.fromkeys(m.lstrip("0") or "0" for m in fig_matches))

    # Bounding boxes for standalone numerals (reference numerals)
    img_w, img_h = _png_dimensions(img_bytes)

    # Collect "FIG"/"Figure" token pixel bboxes
    fig_tokens: list[tuple[int, int, int, int]] = []  # (x_min, x_max, y_min, y_max)
    for ann in annotations[1:]:
        if re.match(r"^(?:FIG|Figure)\.?$", ann["description"], re.IGNORECASE):
            verts = ann.get("boundingPoly", {}).get("vertices", [])
            if len(verts) >= 4:
                xs = [v.get("x", 0) for v in verts]
                ys = [v.get("y", 0) for v in verts]
                fig_tokens.append((min(xs), max(xs), min(ys), max(ys)))
    fig_number_set = set(figure_numbers)

    # Build candidate numerals with pixel bounds for distance checks
    # Each: (bbox_dict, cx, cy, numeral, px_x_min, px_x_max, px_y_min, px_y_max)
    candidates: list[tuple[dict, float, float, str, int, int, int, int]] = []
    for ann in annotations[1:]:
        desc = ann["description"].rstrip("-–—.,;:°")
        if not _NUMERAL_PATTERN.match(desc):
            continue
        numeral = desc.lstrip("0") or "0"
        if numeral == "0":
            continue
        verts = ann.get("boundingPoly", {}).get("vertices", [])
        if len(verts) < 4:
            continue
        xs = [v.get("x", 0) for v in verts]
        ys = [v.get("y", 0) for v in verts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        bbox = {
            "numeral": numeral,
            "x": x_min / img_w,
            "y": y_min / img_h,
            "w": (x_max - x_min) / img_w,
            "h": (y_max - y_min) / img_h,
        }
        candidates.append((bbox, cx, cy, numeral, x_min, x_max, y_min, y_max))

    # For each FIG token, find the closest candidate matching a known figure number.
    # Build a combined bbox (FIG token ∪ number token) as a "figure" label bbox.
    fig_label_bboxes: list[dict] = []
    excluded: set[int] = set()
    for fx_min, fx_max, fy_min, fy_max in fig_tokens:
        fcx, fcy = (fx_min + fx_max) / 2, (fy_min + fy_max) / 2
        best_idx = -1
        best_dist = float("inf")
        for i, (_, cx, cy, numeral, *_px) in enumerate(candidates):
            if i in excluded or numeral not in fig_number_set:
                continue
            dist = (cx - fcx) ** 2 + (cy - fcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            excluded.add(best_idx)
            _, _, _, numeral, nx_min, nx_max, ny_min, ny_max = candidates[best_idx]
            # Union of FIG token and number token
            ux_min = min(fx_min, nx_min)
            ux_max = max(fx_max, nx_max)
            uy_min = min(fy_min, ny_min)
            uy_max = max(fy_max, ny_max)
            fig_label_bboxes.append({
                "numeral": numeral,
                "x": ux_min / img_w,
                "y": uy_min / img_h,
                "w": (ux_max - ux_min) / img_w,
                "h": (uy_max - uy_min) / img_h,
                "type": "figure",
            })

    numeral_bboxes = [bbox for i, (bbox, *_) in enumerate(candidates) if i not in excluded]

    return figure_numbers, numeral_bboxes, fig_label_bboxes


async def _build_figure_map(
    figure_urls: list[str],
) -> tuple[dict[str, int], dict[str, list[dict]]]:
    """Download drawing sheets and OCR via Google Vision.

    Returns (figure_map, numeral_locations) where numeral_locations maps each
    detected numeral to a list of {sheet, x, y, w, h} normalized bounding boxes.
    """
    if len(figure_urls) <= 1:
        return {}, {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Download all sheets (skip cover at index 0)
        sheet_bytes: dict[int, bytes] = {}

        async def _download(idx: int, url: str) -> None:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    sheet_bytes[idx] = resp.content
            except Exception as exc:
                logger.warning("Download failed for sheet %d: %s", idx, exc)

        await asyncio.gather(*[
            _download(idx, url)
            for idx, url in enumerate(figure_urls)
            if idx > 0
        ])

        # OCR all sheets via Google Vision (parallel)
        sheet_figs: dict[int, list[str]] = {}
        sheet_bboxes: dict[int, list[dict]] = {}
        sheet_fig_labels: dict[int, list[dict]] = {}

        async def _ocr_sheet(idx: int) -> None:
            figs, bboxes, fig_labels = await _vision_ocr_sheet(sheet_bytes[idx], client)
            numerals = [b["numeral"] for b in bboxes]
            logger.info("[OCR] sheet %d → figs=%s  numerals=%s", idx, figs, numerals)
            if figs:
                sheet_figs[idx] = figs
            if bboxes:
                sheet_bboxes[idx] = bboxes
            if fig_labels:
                sheet_fig_labels[idx] = fig_labels

        await asyncio.gather(*[_ocr_sheet(idx) for idx in sheet_bytes])

    # Build figure map (first occurrence wins)
    figure_map: dict[str, int] = {}
    for idx in sorted(sheet_figs):
        for fig_num in sheet_figs[idx]:
            if fig_num not in figure_map:
                figure_map[fig_num] = idx

    # Build numeral locations (element numerals + figure labels)
    numeral_locations: dict[str, list[dict]] = {}
    for idx in sorted(sheet_bboxes):
        for bbox in sheet_bboxes[idx]:
            numeral = bbox["numeral"]
            entry = {"sheet": idx, "x": bbox["x"], "y": bbox["y"], "w": bbox["w"], "h": bbox["h"]}
            numeral_locations.setdefault(numeral, []).append(entry)
    for idx in sorted(sheet_fig_labels):
        for bbox in sheet_fig_labels[idx]:
            key = f"FIG. {bbox['numeral']}"
            entry = {"sheet": idx, "x": bbox["x"], "y": bbox["y"], "w": bbox["w"], "h": bbox["h"], "type": "figure"}
            numeral_locations.setdefault(key, []).append(entry)

    return figure_map, numeral_locations


@router.get("/{publication_number}/figure-map")
async def get_figure_map(publication_number: str):
    """OCR patent drawing sheets to map figure numbers to drawing sheet indices.

    Returns {"figure_map": {"1": 1, "2": 2, ...}} where keys are figure numbers
    and values are indices into the figure_urls array.
    """
    candidates = _normalize_pub_number(publication_number)

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        for candidate in candidates:
            url = f"https://patents.google.com/patent/{candidate}/en"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = _parse_patent_html(resp.text, candidate)
                    figure_map, numeral_locations = await _build_figure_map(data.figure_urls)
                    return {
                        "figure_map": figure_map,
                        "numeral_locations": numeral_locations,
                    }
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", candidate, exc)
                continue

    raise HTTPException(
        status_code=404,
        detail=f"Patent not found. Tried: {', '.join(candidates)}",
    )
