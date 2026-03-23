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

# Pattern 1: parenthesized — "camera ( 110 )" or "camera (110a)"
_REF_PAREN = re.compile(
    r"((?:\b[\w-]+\s+){1,5})\(\s*(\d+[a-zA-Z]?)\s*\)"
)
# Pattern 2: bare — "camera 110" or "layers 602"
# Requires a letter-word immediately before the number, and the number is 1-5 digits
_REF_BARE = re.compile(
    r"((?:\b[a-zA-Z][\w-]*\s+){1,5})"
    r"(\d{1,5}[a-zA-Z]?)"
    r"(?=[\s,;.\)\]]|$)"
)
# Contexts to skip for bare numerals (FIG. 1, claim 2, step 3, etc.)
_BARE_SKIP = re.compile(
    r"(?:fig\.?|figure|claim|step|page|table|example|section"
    r"|mm|cm|km|m|ms|nm|μm|hz|khz|mhz|ghz|mb|gb|kb|percent|%"
    r"|length|stride|size|width|height|depth|count|number|about|approximately|least|than"
    r"|version|v|no|nos|vol|chapter|paragraph|col|row|eq|equ)\s*$",
    re.I,
)


def _extract_reference_numerals(data: PatentData) -> list[dict]:
    """Extract reference numeral → label mappings from patent text.

    Scans abstract, description, and claims for patterns like "camera ( 110 )"
    and returns a deduplicated list sorted by numeral.
    """
    # Collect all text
    parts = [data.abstract]
    for section in data.description:
        parts.extend(p.text for p in section.paragraphs)
    for claim in data.claims:
        parts.append(claim.text)
    all_text = " ".join(parts)

    # Find all (context, numeral) pairs from both patterns
    num_labels: defaultdict[str, list[str]] = defaultdict(list)

    def _add_match(context: str, numeral: str) -> None:
        # Clean label: take last 1-4 words, strip leading/trailing stop words
        words = context.split()[-4:]
        while words and words[0].lower() in _STOP_WORDS:
            words = words[1:]
        while words and words[-1].lower() in _STOP_WORDS:
            words = words[:-1]
        if not words:
            return
        label = " ".join(words).lower()
        # Strip common prepositional noise from label
        label = re.sub(r"^.*?\b(?:of|from|on|via|into|over)\s+(?:a|an|the)\s+", "", label)
        if not label:
            return
        # Skip likely method step numbers (single word that's a verb/gerund)
        if len(words) == 1 and words[0].endswith("ing"):
            return
        # Skip numerals that are clearly not references (units, dimensions like 3D)
        if re.match(r"^\d+[dD]$", numeral):  # 2D, 3D, 4D
            return
        num_labels[numeral].append(label)

    # Pattern 1: parenthesized references — "camera ( 110 )"
    for match in _REF_PAREN.finditer(all_text):
        _add_match(match.group(1).strip(), match.group(2))

    # Pattern 2: bare references — "camera 110"
    for match in _REF_BARE.finditer(all_text):
        context = match.group(1).strip()
        if _BARE_SKIP.search(context):
            continue
        _add_match(context, match.group(2))

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


# ── Figure map via Google Vision OCR ──────────────────────────────────────

_FIG_PATTERN = re.compile(r"Fig(?:ure)?\s*[.,;:]?\s*(\d+[a-zA-Z]?)", re.IGNORECASE)
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
    numeral_bboxes: list[dict] = []
    for ann in annotations[1:]:
        desc = ann["description"]
        if not _NUMERAL_PATTERN.match(desc):
            continue
        numeral = desc.lstrip("0") or "0"
        verts = ann.get("boundingPoly", {}).get("vertices", [])
        if len(verts) < 4:
            continue
        xs = [v.get("x", 0) for v in verts]
        ys = [v.get("y", 0) for v in verts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        numeral_bboxes.append({
            "numeral": numeral,
            "x": x_min / img_w,
            "y": y_min / img_h,
            "w": (x_max - x_min) / img_w,
            "h": (y_max - y_min) / img_h,
        })

    return figure_numbers, numeral_bboxes


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

        async def _ocr_sheet(idx: int) -> None:
            figs, bboxes = await _vision_ocr_sheet(sheet_bytes[idx], client)
            if figs:
                sheet_figs[idx] = figs
            if bboxes:
                sheet_bboxes[idx] = bboxes

        await asyncio.gather(*[_ocr_sheet(idx) for idx in sheet_bytes])

    # Build figure map (first occurrence wins)
    figure_map: dict[str, int] = {}
    for idx in sorted(sheet_figs):
        for fig_num in sheet_figs[idx]:
            if fig_num not in figure_map:
                figure_map[fig_num] = idx

    # Build numeral locations
    numeral_locations: dict[str, list[dict]] = {}
    for idx in sorted(sheet_bboxes):
        for bbox in sheet_bboxes[idx]:
            numeral = bbox["numeral"]
            entry = {"sheet": idx, "x": bbox["x"], "y": bbox["y"], "w": bbox["w"], "h": bbox["h"]}
            numeral_locations.setdefault(numeral, []).append(entry)

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
