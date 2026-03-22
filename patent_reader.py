"""Patent Reader API — fetch and return structured patent data from Google Patents."""

import asyncio
import io
import logging
import re
from dataclasses import dataclass, field, asdict

from collections import Counter, defaultdict

import httpx
import numpy as np
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException
from PIL import Image

import pytesseract

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


# ── Figure map via OCR ───────────────────────────────────────────────────

# Match "Fig. 11" but also OCR misreads like "Fig. ii", "Hig. l3", "Fig, 1I"
_FIG_PATTERN = re.compile(r"[FfHh]ig(?:ure)?\s*[.,;:]?\s*([0-9ilIoO]+[a-zA-Z]?)", re.IGNORECASE)


def _fix_ocr_digits(s: str) -> str | None:
    """Normalize common OCR misreads in figure numbers: i/l/I→1, O/o→0."""
    fixed = s.translate(str.maketrans("ilIoO", "11100"))
    # Must be a valid figure number (starts with digit, not all zeros)
    if re.match(r"^\d+[a-zA-Z]?$", fixed) and fixed.lstrip("0"):
        return fixed.lstrip("0")
    return None


def _try_ocr_rotations(img: Image.Image) -> list[str]:
    """Try OCR on an image at 3 rotations (0°, -90°, 90°), return figure numbers found."""
    for angle in [0, -90, 90]:
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        text = pytesseract.image_to_string(rotated, config="--psm 11")
        raw_matches = _FIG_PATTERN.findall(text)
        if raw_matches:
            fixed = set()
            for m in raw_matches:
                f = _fix_ocr_digits(m)
                if f:
                    fixed.add(f)
            if fixed:
                return list(fixed)
    return []


def _remove_line_art(img: Image.Image) -> Image.Image:
    """Binarize a grayscale image and remove horizontal/vertical runs > 50px."""
    arr = np.array(img)
    binary = (arr < 128).astype(np.uint8)
    cleaned = binary.copy()
    min_line = 50

    for y in range(cleaned.shape[0]):
        row = cleaned[y]
        start = None
        for x in range(len(row)):
            if row[x] == 1:
                if start is None:
                    start = x
            else:
                if start is not None:
                    if x - start > min_line:
                        cleaned[y, start:x] = 0
                    start = None
        if start is not None and len(row) - start > min_line:
            cleaned[y, start:] = 0

    for x in range(cleaned.shape[1]):
        col = cleaned[:, x]
        start = None
        for y in range(len(col)):
            if col[y] == 1:
                if start is None:
                    start = y
            else:
                if start is not None:
                    if y - start > min_line:
                        cleaned[start:y, x] = 0
                    start = None
        if start is not None and len(col) - start > min_line:
            cleaned[start:, x] = 0

    return Image.fromarray(((1 - cleaned) * 255).astype(np.uint8))


def _ocr_figure_numbers(img_bytes: bytes, mode: str = "raw") -> list[str]:
    """Extract figure numbers from a patent drawing sheet via OCR.

    mode="raw": try raw grayscale first, then line-removed fallback.
    mode="line_removed": only try line-removed (used for outlier retry).
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    if mode == "raw":
        results = _try_ocr_rotations(img)
        if results:
            return results

    line_removed = _remove_line_art(img)
    return _try_ocr_rotations(line_removed)


def _fig_numeric_part(fig_num: str) -> int:
    """Extract the numeric part of a figure number (e.g. '17A' → 17)."""
    m = re.match(r"(\d+)", fig_num)
    return int(m.group(1)) if m else 0


def _find_outlier_sheets(
    sheet_results: dict[int, list[str]],
) -> set[int]:
    """Identify sheets whose figure numbers are statistical outliers (IQR method)."""
    all_nums = []
    for figs in sheet_results.values():
        all_nums.extend(_fig_numeric_part(f) for f in figs)

    if len(all_nums) < 4:
        return set()

    all_nums.sort()
    q1 = all_nums[len(all_nums) // 4]
    q3 = all_nums[3 * len(all_nums) // 4]
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    outlier_sheets = set()
    for idx, figs in sheet_results.items():
        for f in figs:
            if _fig_numeric_part(f) > upper_fence:
                outlier_sheets.add(idx)
    return outlier_sheets


async def _build_figure_map(figure_urls: list[str]) -> dict[str, int]:
    """Download drawing sheets and OCR them to map figure numbers to URL indices.

    Strategy:
    1. Raw OCR all sheets (parallel).
    2. Cross-sheet outlier detection (IQR).
    3. Re-OCR outlier sheets with line-removed; re-OCR empty sheets with line-removed.
    4. Discard any results still flagged as outliers.
    """
    if len(figure_urls) <= 1:
        return {}

    # Download all sheets (skip cover at index 0), keep bytes for potential retry
    sheet_bytes: dict[int, bytes] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
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

    # Phase 1: Raw OCR all sheets
    sheet_results: dict[int, list[str]] = {}
    empty_sheets: set[int] = set()

    async def _ocr_sheet(idx: int, mode: str) -> list[str]:
        try:
            return await asyncio.to_thread(
                _ocr_figure_numbers, sheet_bytes[idx], mode,
            )
        except Exception as exc:
            logger.warning("OCR failed for sheet %d: %s", idx, exc)
            return []

    raw_tasks = {idx: _ocr_sheet(idx, "raw") for idx in sheet_bytes}
    raw_results = await asyncio.gather(*raw_tasks.values())
    for idx, figs in zip(raw_tasks.keys(), raw_results):
        if figs:
            sheet_results[idx] = figs
        else:
            empty_sheets.add(idx)

    # Phase 2: Find outlier sheets
    outlier_sheets = _find_outlier_sheets(sheet_results)

    # Phase 3: Re-OCR outlier + empty sheets with line-removed
    retry_sheets = outlier_sheets | empty_sheets
    if retry_sheets:
        retry_tasks = {idx: _ocr_sheet(idx, "line_removed") for idx in retry_sheets}
        retry_results = await asyncio.gather(*retry_tasks.values())
        for idx, figs in zip(retry_tasks.keys(), retry_results):
            sheet_results[idx] = figs  # replace (outlier) or fill (empty)

    # Phase 4: Final outlier check — discard any remaining outliers
    if outlier_sheets:
        still_outliers = _find_outlier_sheets(sheet_results)
        for idx in still_outliers:
            sheet_results.pop(idx, None)

    # Build map (first occurrence wins)
    figure_map: dict[str, int] = {}
    for idx in sorted(sheet_results):
        for fig_num in sheet_results[idx]:
            if fig_num not in figure_map:
                figure_map[fig_num] = idx

    return figure_map


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
                    figure_map = await _build_figure_map(data.figure_urls)
                    return {"figure_map": figure_map}
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", candidate, exc)
                continue

    raise HTTPException(
        status_code=404,
        detail=f"Patent not found. Tried: {', '.join(candidates)}",
    )
