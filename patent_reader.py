"""Patent Reader API — fetch and return structured patent data from Google Patents."""

import asyncio
import base64
import logging
import os
import re
import struct
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher

from collections import Counter, defaultdict

import httpx
from bs4 import BeautifulSoup, Tag
from fastapi import APIRouter, HTTPException

from core.nlp_models import nlp as spacy_nlp
from patent_cache import get_cached, set_cached

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patents", tags=["patents"])


# ── Structured types ──────────────────────────────────────────────────────


@dataclass
class ClaimLimitation:
    text: str
    depth: int
    children: list["ClaimLimitation"] = field(default_factory=list)


@dataclass
class PatentClaim:
    number: int
    text: str  # flat text (backwards compat)
    depends_on: int | None  # None = independent
    type: str  # "independent" | "dependent"
    limitations: list[ClaimLimitation] = field(default_factory=list)


@dataclass
class PatentParagraph:
    text: str
    number: str | None = None  # e.g. "0001", absent for some patents
    col: int | None = None      # column number (older patents without para numbers)
    line: int | None = None     # start line number
    end_col: int | None = None  # end column number
    end_line: int | None = None # end line number
    line_breaks: list[dict] | None = None  # [{offset, col, line}, ...] for selection → col/line mapping


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


def _patent_data_from_dict(d: dict) -> PatentData:
    """Reconstruct a PatentData from a dict (e.g. from JSON cache)."""
    def _limitation(lim: dict) -> ClaimLimitation:
        return ClaimLimitation(
            text=lim["text"],
            depth=lim["depth"],
            children=[_limitation(c) for c in lim.get("children", [])],
        )

    return PatentData(
        title=d["title"],
        patent_number=d["patent_number"],
        filing_date=d["filing_date"],
        publication_date=d["publication_date"],
        inventors=d["inventors"],
        assignee=d["assignee"],
        classifications=d["classifications"],
        abstract=d["abstract"],
        claims=[
            PatentClaim(
                number=c["number"],
                text=c["text"],
                depends_on=c["depends_on"],
                type=c["type"],
                limitations=[_limitation(lim) for lim in c.get("limitations", [])],
            )
            for c in d["claims"]
        ],
        description=[
            PatentSection(
                heading=s["heading"],
                paragraphs=[
                    PatentParagraph(
                        text=p["text"],
                        number=p.get("number"),
                        col=p.get("col"),
                        line=p.get("line"),
                        end_col=p.get("end_col"),
                        end_line=p.get("end_line"),
                        line_breaks=p.get("line_breaks"),
                    )
                    for p in s["paragraphs"]
                ],
            )
            for s in d["description"]
        ],
        pdf_url=d.get("pdf_url", ""),
        figure_urls=d.get("figure_urls", []),
        priority_date=d.get("priority_date", ""),
        patent_citations=[PatentCitation(**c) for c in d.get("patent_citations", [])],
        cited_by=[PatentCitation(**c) for c in d.get("cited_by", [])],
        non_patent_citations=[NonPatentCitation(**c) for c in d.get("non_patent_citations", [])],
        family_applications=[FamilyApplication(**c) for c in d.get("family_applications", [])],
        country_status=[CountryStatus(**c) for c in d.get("country_status", [])],
        legal_events=[LegalEvent(**e) for e in d.get("legal_events", [])],
        similar_documents=[SimilarDocument(**s) for s in d.get("similar_documents", [])],
    )


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


def _parse_claim_limitations(claim_div: Tag) -> list[ClaimLimitation]:
    """Parse nested claim-text divs into a limitation tree.

    Google Patents encodes claim structure as nested <div class="claim-text">
    elements. Each nesting level represents a deeper limitation (preamble →
    body elements → sub-elements).
    """
    def _get_direct_text(node: Tag) -> str:
        """Get text from a claim-text div, excluding nested claim-text children."""
        parts = []
        for child in node.children:
            if isinstance(child, str):
                parts.append(child)
            elif isinstance(child, Tag):
                if "claim-text" in (child.get("class") or []):
                    continue  # skip nested claim-text
                parts.append(child.get_text(separator=" "))
        text = " ".join(parts).strip()
        # Strip leading claim number ("1.", "1 .")
        text = re.sub(r"^\d+\s*\.\s*", "", text).strip()
        return text

    def _walk(node: Tag, depth: int) -> list[ClaimLimitation]:
        results = []
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            if "claim-text" not in (child.get("class") or []):
                continue
            text = _get_direct_text(child)
            children = _walk(child, depth + 1)
            if text or children:
                results.append(ClaimLimitation(text=text, depth=depth, children=children))
        return results

    return _walk(claim_div, 0)


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

    # Claims — parse with dependency info and limitation tree
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

                # Parse nested claim-text divs into limitation tree
                limitations = _parse_claim_limitations(claim_div)

                claims.append(PatentClaim(
                    number=num,
                    text=text,
                    depends_on=depends_on,
                    type="dependent" if depends_on is not None else "independent",
                    limitations=limitations,
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


# ── Col/line extraction for older patents ─────────────────────────────────


def _col_line_normalize(text: str) -> str:
    """Normalize text for fuzzy comparison between HTML and PDF text."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("?", "")  # PDF ligature placeholder
    return text.strip()


def _extract_pdf_lines(pdf_bytes: bytes) -> list[dict]:
    """Extract all text lines with col/line numbers from a patent PDF."""
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    all_lines = []

    for pg_idx in range(len(doc)):
        page = doc[pg_idx]
        text = page.get_text()

        if "Sheet" in text and " of " in text:
            continue

        data = page.get_text("dict")
        width = page.rect.width

        text_lines = []
        for block in data["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                full_text = "".join(s["text"] for s in line["spans"]).strip()
                if not full_text:
                    continue
                bbox = line["bbox"]
                avg_size = sum(s["size"] for s in line["spans"]) / len(line["spans"])
                text_lines.append({
                    "x0": bbox[0], "y0": bbox[1],
                    "x1": bbox[2], "y1": bbox[3],
                    "text": full_text, "size": avg_size,
                })

        if not text_lines:
            continue

        # Find gutter line markers (multiples of 5, small font, mid-page x)
        candidate_markers = []
        marker_indices = set()
        for i, tl in enumerate(text_lines):
            t = tl["text"].replace(",", "").strip()
            if t.isdigit():
                num = int(t)
                if 5 <= num <= 70 and num % 5 == 0:
                    if tl["size"] < 8 and (width * 0.35 < tl["x0"] < width * 0.65):
                        candidate_markers.append({"line_num": num, "x": tl["x0"], "y": tl["y0"]})
                        marker_indices.add(i)

        if not candidate_markers:
            continue

        gutter_x = sum(m["x"] for m in candidate_markers) / len(candidate_markers)
        markers_sorted = sorted(candidate_markers, key=lambda m: m["y"])

        # Detect column numbers from header
        col1_num = None
        col2_num = None
        for tl in text_lines:
            if tl["y0"] < 80 and tl["text"].strip().isdigit():
                num = int(tl["text"].strip())
                if num < 30:
                    if tl["x0"] < gutter_x:
                        col1_num = num
                    else:
                        col2_num = num

        # Assign lines to columns with interpolated line numbers
        for i, tl in enumerate(text_lines):
            if i in marker_indices or tl["y0"] < 75:
                continue

            col_num = col1_num if tl["x0"] < gutter_x else col2_num
            if col_num is None:
                continue

            y = tl["y0"]
            lower = upper = None
            for m in markers_sorted:
                if m["y"] <= y:
                    lower = m
                elif upper is None:
                    upper = m

            line_est = None
            if lower and upper:
                frac = (y - lower["y"]) / (upper["y"] - lower["y"])
                line_est = round(lower["line_num"] + frac * (upper["line_num"] - lower["line_num"]))
            elif lower and len(markers_sorted) >= 2:
                m1, m2 = markers_sorted[-2], markers_sorted[-1]
                rate = (m2["line_num"] - m1["line_num"]) / (m2["y"] - m1["y"])
                line_est = round(lower["line_num"] + (y - lower["y"]) * rate)
            elif upper and len(markers_sorted) >= 2:
                m1, m2 = markers_sorted[0], markers_sorted[1]
                rate = (m2["line_num"] - m1["line_num"]) / (m2["y"] - m1["y"])
                line_est = round(upper["line_num"] - (upper["y"] - y) * rate)

            if line_est is not None:
                all_lines.append({"col": col_num, "line": line_est, "text": tl["text"]})

    all_lines.sort(key=lambda x: (x["col"], x["line"]))
    return all_lines


def _find_col_line_match(
    para_text: str,
    pdf_lines: list[dict],
    search_from: int = 0,
    min_ratio: float = 0.5,
) -> tuple[dict | None, int]:
    """Find the col/line range in pdf_lines that best matches a paragraph.

    Uses sequential cursor + tail-similarity end detection + lookahead.
    Returns (match_dict, next_search_from).
    """
    para_norm = _col_line_normalize(para_text)
    if len(para_norm) < 10:
        return None, search_from

    # Find START — match first ~60 chars
    search_start = para_norm[:60]
    best_start_idx = None
    best_start_score = 0.0

    for i in range(search_from, len(pdf_lines)):
        pl_norm = _col_line_normalize(pdf_lines[i]["text"])
        compare_len = min(len(search_start), len(pl_norm))
        if compare_len < 5:
            continue
        ratio = SequenceMatcher(None, search_start[:compare_len], pl_norm[:compare_len]).ratio()
        if ratio > best_start_score and ratio >= min_ratio:
            best_start_score = ratio
            best_start_idx = i

    if best_start_idx is None:
        return None, search_from

    # Find END — accumulate lines, check tail similarity
    para_tail = para_norm[-50:]
    concat = _col_line_normalize(pdf_lines[best_start_idx]["text"])
    end_idx = best_start_idx
    best_end_idx = best_start_idx
    best_tail_score = 0.0

    est_lines = max(2, len(para_norm) // 40)
    max_scan = best_start_idx + est_lines + 15

    for j in range(best_start_idx + 1, min(max_scan, len(pdf_lines))):
        next_norm = _col_line_normalize(pdf_lines[j]["text"])

        if pdf_lines[j]["col"] != pdf_lines[j - 1]["col"]:
            if best_tail_score > 0.7:
                break
            # Paragraph spans columns — keep going

        concat += " " + next_norm
        end_idx = j

        concat_tail = concat[-50:]
        tail_score = SequenceMatcher(None, para_tail, concat_tail).ratio()

        if tail_score > best_tail_score:
            best_tail_score = tail_score
            best_end_idx = j

        if tail_score > 0.7 and len(concat) >= len(para_norm) * 0.7:
            break

        if len(concat) > len(para_norm) * 1.5:
            break

    # Lookahead: catch orphaned trailing words (e.g., "detail." alone on its own line)
    for k in range(best_end_idx + 1, min(best_end_idx + 3, max_scan, len(pdf_lines))):
        peek_norm = _col_line_normalize(pdf_lines[k]["text"])
        peek_concat = concat + " " + peek_norm if k > end_idx else \
            " ".join(_col_line_normalize(pdf_lines[m]["text"]) for m in range(best_start_idx, k + 1))
        peek_tail = peek_concat[-50:]
        peek_score = SequenceMatcher(None, para_tail, peek_tail).ratio()
        if peek_score > best_tail_score:
            best_tail_score = peek_score
            best_end_idx = k
            concat = peek_concat
        else:
            break

    start_line = pdf_lines[best_start_idx]
    end_line = pdf_lines[best_end_idx]

    return {
        "start_col": start_line["col"],
        "start_line": start_line["line"],
        "end_col": end_line["col"],
        "end_line": end_line["line"],
        "match_start_idx": best_start_idx,
        "match_end_idx": best_end_idx,
    }, best_end_idx + 1


def _compute_line_breaks(
    para_text: str,
    pdf_lines: list[dict],
    start_idx: int,
    end_idx: int,
) -> list[dict]:
    """Compute character offsets in para_text where each PDF line boundary falls.

    For each matched PDF line, finds where its text begins in the paragraph
    using sequential substring search. Falls back to estimation if not found.
    """
    para_lower = para_text.lower()
    line_breaks: list[dict] = []
    search_pos = 0

    for i in range(start_idx, end_idx + 1):
        pdf_line = pdf_lines[i]
        raw = pdf_line["text"]
        col = pdf_line["col"]
        ln = pdf_line["line"]

        found = False
        for chunk_len in (20, 12, 8, 5):
            chunk = raw[:chunk_len].strip().lower()
            if len(chunk) < 3:
                continue
            idx = para_lower.find(chunk, search_pos)
            if idx != -1:
                line_breaks.append({"offset": idx, "col": col, "line": ln})
                search_pos = idx + 1
                found = True
                break

        if not found:
            if line_breaks:
                avg_gap = line_breaks[-1]["offset"] // max(1, len(line_breaks))
                est = min(search_pos + avg_gap, len(para_text) - 1)
                line_breaks.append({"offset": est, "col": col, "line": ln})
            else:
                line_breaks.append({"offset": 0, "col": col, "line": ln})

    return line_breaks


def _apply_col_line_locations(data: PatentData, pdf_bytes: bytes) -> None:
    """Assign col/line locations to description paragraphs using PDF extraction."""
    pdf_lines = _extract_pdf_lines(pdf_bytes)
    if not pdf_lines:
        return
    cursor = 0
    for section in data.description:
        for para in section.paragraphs:
            match, cursor = _find_col_line_match(para.text, pdf_lines, search_from=cursor)
            if match:
                para.col = match["start_col"]
                para.line = match["start_line"]
                para.end_col = match["end_col"]
                para.end_line = match["end_line"]
                para.line_breaks = _compute_line_breaks(
                    para.text, pdf_lines,
                    match["match_start_idx"], match["match_end_idx"],
                )


# ── Shared fetch + in-memory cache ────────────────────────────────────────

_patent_cache: dict[str, PatentData] = {}
_patent_locks: dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()


async def _get_patent_data(publication_number: str) -> PatentData:
    """Fetch and parse a patent, returning a cached result if available.

    L1 = in-memory dict (hot path, lost on restart).
    L2 = SQLite patent_cache table (survives restarts).
    L3 = Google Patents fetch.
    """
    candidates = _normalize_pub_number(publication_number)

    # L1: in-memory check before acquiring any lock
    for candidate in candidates:
        if candidate in _patent_cache:
            return _patent_cache[candidate]

    cache_key = candidates[0]
    async with _locks_lock:
        if cache_key not in _patent_locks:
            _patent_locks[cache_key] = asyncio.Lock()
        lock = _patent_locks[cache_key]

    async with lock:
        # Double-check L1 after acquiring lock
        for candidate in candidates:
            if candidate in _patent_cache:
                return _patent_cache[candidate]

        # L2: SQLite persistent cache
        for candidate in candidates:
            cached = await get_cached(candidate, "patent")
            if cached:
                data = _patent_data_from_dict(cached)
                _patent_cache[candidate] = data
                return data

        # L3: fetch from Google Patents
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            for candidate in candidates:
                url = f"https://patents.google.com/patent/{candidate}/en"
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = _parse_patent_html(resp.text, candidate)
                        # Col/line extraction for patents without paragraph numbers
                        all_paras = [p for s in data.description for p in s.paragraphs]
                        if all_paras and all(p.number is None for p in all_paras):
                            # Use /pdfs/ URL (original patent) rather than citation_pdf_url
                            # (which may include reexam certificates that change page layout)
                            bare_number = re.sub(r"[A-Z]$", "", candidate.replace("US", ""))
                            col_line_pdf_url = f"https://patentimages.storage.googleapis.com/pdfs/US{bare_number}.pdf"
                            try:
                                pdf_resp = await client.get(col_line_pdf_url)
                                if pdf_resp.status_code == 200:
                                    _apply_col_line_locations(data, pdf_resp.content)
                            except httpx.HTTPError:
                                pass  # non-fatal
                        _patent_cache[candidate] = data
                        await set_cached(candidate, "patent", asdict(data))
                        return data
                except httpx.HTTPError as exc:
                    logger.warning("Failed to fetch %s: %s", candidate, exc)
                    continue

        raise HTTPException(
            status_code=404,
            detail=f"Patent not found. Tried: {', '.join(candidates)}",
        )


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get("/{publication_number}")
async def get_patent(publication_number: str):
    """Fetch and return structured patent data from Google Patents."""
    data = await _get_patent_data(publication_number)
    return asdict(data)


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
    """Extract reference numeral → label mappings and highlight positions from a patent."""
    data = await _get_patent_data(publication_number)
    key = data.patent_number
    cached = await get_cached(key, "reference_numerals")
    if cached:
        return cached
    result = _extract_reference_numerals(data)
    await set_cached(key, "reference_numerals", result)
    return result


# ── Claim element extraction ────────────────────────────────────────────


_CLAIM_ELEMENT_SKIP = frozenset(
    "claim claims step steps method system apparatus device "
    "embodiment embodiments example examples means way "
    "communication service services use case".split()
)


def _extract_claim_elements(data: PatentData) -> dict:
    """Extract claim element introductions and back-references with group IDs.

    For each claim, identifies noun phrases introduced with "a"/"an" (or bare)
    and their back-references with "the"/"said". Groups matching introductions
    and references so the frontend can color-code them.

    Walks dependency chains so dependent claims inherit parent introductions.
    """
    claims_map = {c.number: c for c in data.claims}

    # Pass 1: extract NPs per claim and build introduction maps
    claim_nps: dict[int, list[dict]] = {}
    claim_intros: dict[int, dict[str, int]] = {}
    group_counter = 0

    for claim in data.claims:
        doc = spacy_nlp(claim.text)
        nps = []

        for chunk in doc.noun_chunks:
            det = None
            det_token = None
            for tok in chunk:
                if tok.dep_ == "det":
                    det = tok.text.lower()
                    det_token = tok
                    break

            if det_token:
                np_start_offset = det_token.idx + len(det_token.text_with_ws) - chunk.start_char
                np_text = chunk.text[np_start_offset:].strip().lower()
            else:
                np_text = chunk.text.strip().lower()

            if not np_text:
                continue

            if det in ("a", "an"):
                role = "introduction"
            elif det in ("the", "said"):
                role = "reference"
            elif det is None:
                role = "bare"
            else:
                continue

            # Skip structural/noise words for bare NPs
            if role == "bare" and np_text in _CLAIM_ELEMENT_SKIP:
                continue

            nps.append({
                "start": chunk.start_char,
                "end": chunk.end_char,
                "np_text": np_text,
                "role": role,
            })

        claim_nps[claim.number] = nps

        intros: dict[str, int] = {}
        for np in nps:
            if np["role"] in ("introduction", "bare") and np["np_text"] not in intros:
                intros[np["np_text"]] = group_counter
                group_counter += 1
        claim_intros[claim.number] = intros

    # Pass 2: collect inherited introductions via dependency chains
    inherited_cache: dict[int, dict[str, int]] = {}

    def _collect_intros(claim_num: int, visited: set[int] | None = None) -> dict[str, int]:
        if claim_num in inherited_cache:
            return inherited_cache[claim_num]
        if visited is None:
            visited = set()
        if claim_num in visited:
            return {}
        visited.add(claim_num)
        claim = claims_map.get(claim_num)
        if not claim:
            return {}
        result = dict(claim_intros.get(claim_num, {}))
        if claim.depends_on is not None:
            parent = _collect_intros(claim.depends_on, visited)
            for np_text, gid in parent.items():
                if np_text not in result:
                    result[np_text] = gid
        inherited_cache[claim_num] = result
        return result

    # Pass 3: build output spans with group IDs
    claim_elements = []
    all_groups: dict[int, dict] = {}

    for claim in data.claims:
        all_intros = _collect_intros(claim.number)
        spans = []

        for np in claim_nps[claim.number]:
            if np["role"] in ("introduction", "bare"):
                gid = claim_intros[claim.number].get(np["np_text"])
                if gid is not None:
                    spans.append({
                        "start": np["start"],
                        "end": np["end"],
                        "group_id": gid,
                        "np_text": np["np_text"],
                        "role": np["role"],
                    })
                    if gid not in all_groups:
                        all_groups[gid] = {
                            "group_id": gid,
                            "np_text": np["np_text"],
                            "introduced_in": claim.number,
                        }
            elif np["role"] == "reference":
                gid = all_intros.get(np["np_text"])
                if gid is not None:
                    spans.append({
                        "start": np["start"],
                        "end": np["end"],
                        "group_id": gid,
                        "np_text": np["np_text"],
                        "role": "reference",
                    })

        claim_elements.append({
            "claim_number": claim.number,
            "spans": spans,
        })

    groups = sorted(all_groups.values(), key=lambda g: g["group_id"])
    return {"claim_elements": claim_elements, "groups": groups}


@router.get("/{publication_number}/claim-elements")
async def get_claim_elements(publication_number: str):
    """Extract claim element introductions and references with group IDs."""
    data = await _get_patent_data(publication_number)
    key = data.patent_number
    cached = await get_cached(key, "claim_elements")
    if cached:
        return cached
    result = _extract_claim_elements(data)
    await set_cached(key, "claim_elements", result)
    return result


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


_figure_map_locks: dict[str, asyncio.Lock] = {}


@router.get("/{publication_number}/figure-map")
async def get_figure_map(publication_number: str):
    """OCR patent drawing sheets to map figure numbers to drawing sheet indices."""
    data = await _get_patent_data(publication_number)
    key = data.patent_number

    # Fast path: persistent cache
    cached = await get_cached(key, "figure_map")
    if cached:
        return cached

    # Slow path with lock to prevent duplicate OCR runs
    async with _locks_lock:
        if key not in _figure_map_locks:
            _figure_map_locks[key] = asyncio.Lock()
        lock = _figure_map_locks[key]

    async with lock:
        # Double-check after acquiring lock
        cached = await get_cached(key, "figure_map")
        if cached:
            return cached
        figure_map, numeral_locations = await _build_figure_map(data.figure_urls)
        result = {
            "figure_map": figure_map,
            "numeral_locations": numeral_locations,
        }
        await set_cached(key, "figure_map", result)
        return result
