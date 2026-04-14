"""Fetch and parse patents from Google Patents."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


@dataclass
class PatentFetchResult:
    success: bool
    publication_number: str
    title: str = ""
    abstract_snippet: str = ""
    claim_count: int = 0
    file_path: str = ""
    error: str = ""


def normalize_publication_number(raw: str) -> list[str]:
    """Normalize a publication number from an office action into Google Patents format.

    Handles formats like:
      "US 2022/0075747"
      "US 2022/0075747 A1"
      "US20220075747A1"
      "US 11,234,567 B2"
      "US11234567B2"

    Returns a list of candidates to try (with kind code variants if none given).
    """
    cleaned = raw.strip().upper()
    # Remove commas
    cleaned = cleaned.replace(",", "")

    # Extract country code (2 letters), number, and optional kind code
    m = re.match(
        r"([A-Z]{2})\s*"           # country code
        r"(\d[\d\s/]*\d)"          # number (may contain spaces, slashes)
        r"\s*([A-Z]\d?)?\s*$",     # optional kind code
        cleaned,
    )
    if not m:
        # Fallback: strip to alphanumeric only
        return [re.sub(r"[^A-Z0-9]", "", cleaned)]

    country = m.group(1)
    number = re.sub(r"[\s/]", "", m.group(2))
    kind = m.group(3)

    base = f"{country}{number}"

    if kind:
        # For WO patents, also try without kind code (Google Patents often
        # indexes them as just the base number, e.g. WO2010089455 not WO2010089455A2)
        if country == "WO":
            return [f"{base}{kind}", base]
        return [f"{base}{kind}"]

    # No kind code provided — try common variants
    if country == "WO":
        return [base, f"{base}A1", f"{base}A2", f"{base}A3"]
    return [f"{base}A1", f"{base}B1", f"{base}B2", f"{base}A"]


async def fetch_patent(publication_number: str, workspace: Path) -> PatentFetchResult:
    """Fetch a patent from Google Patents and save as markdown.

    Args:
        publication_number: Raw publication number (e.g. "US 2022/0075747 A1")
        workspace: Session workspace directory

    Returns:
        PatentFetchResult with success status and metadata
    """
    candidates = normalize_publication_number(publication_number)
    normalized = candidates[0]  # Use first candidate as the canonical name

    # Validate normalized name is safe for filesystem
    if not re.match(r"^[A-Z0-9]+$", normalized):
        return PatentFetchResult(
            success=False,
            publication_number=publication_number,
            error="Invalid publication number format",
        )

    # Check cache
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    cache_path = input_dir / f"{normalized}.md"
    if cache_path.exists():
        # Parse cached file for metadata
        text = cache_path.read_text()
        title = ""
        claim_count = 0
        for line in text.split("\n"):
            if line.startswith("# ") and not title:
                title = line[2:].strip()
            if line.startswith("## Claims"):
                # Count claim lines
                claim_count = text.count("\n### Claim ")
                break
        return PatentFetchResult(
            success=True,
            publication_number=normalized,
            title=title,
            abstract_snippet="(cached)",
            claim_count=claim_count,
            file_path=str(cache_path.relative_to(workspace)),
        )

    # Try each candidate URL
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        for candidate in candidates:
            url = f"https://patents.google.com/patent/{candidate}/en"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    normalized = candidate
                    cache_path = input_dir / f"{normalized}.md"
                    return await _parse_and_save(
                        resp.text, normalized, workspace, cache_path,
                    )
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", candidate, exc)
                continue

    return PatentFetchResult(
        success=False,
        publication_number=normalized,
        error=f"Could not fetch patent. Tried: {', '.join(candidates)}",
    )


async def _parse_and_save(
    html: str,
    pub_number: str,
    workspace: Path,
    cache_path: Path,
) -> PatentFetchResult:
    """Parse Google Patents HTML and save as markdown."""
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_meta = soup.find("meta", {"name": "DC.title"})
    title = title_meta.get("content", "").strip() if title_meta else ""
    title = title or pub_number

    # Abstract
    abstract_text = ""
    abstract_section = soup.find("section", itemprop="abstract")
    if abstract_section:
        content_div = abstract_section.find("div", itemprop="content")
        if content_div:
            abstract_text = content_div.get_text(separator=" ", strip=True)

    # Description
    description_parts: list[str] = []
    desc_section = soup.find("section", itemprop="description")
    if desc_section:
        content_div = desc_section.find("div", itemprop="content")
        if content_div:
            desc_list = content_div.find("ul", class_="description")
            if desc_list:
                for child in desc_list.children:
                    if not isinstance(child, Tag):
                        continue
                    if child.name == "heading":
                        description_parts.append(f"\n### {child.get_text(strip=True)}\n")
                    elif child.name == "li":
                        num = child.get("num", "")
                        text = child.get_text(separator=" ", strip=True)
                        prefix = f"[{num}] " if num else ""
                        description_parts.append(f"{prefix}{text}")

    # Claims — find all numbered claim divs (they nest arbitrarily)
    claim_texts: list[str] = []
    claims_section = soup.find("section", itemprop="claims")
    if claims_section:
        claims_div = claims_section.find("div", class_="claims")
        if claims_div:
            for claim_div in claims_div.find_all("div", class_="claim", attrs={"num": True}):
                num = claim_div["num"].lstrip("0") or "?"
                text = claim_div.get_text(separator=" ", strip=True)
                # Strip leading claim number — Google Patents embeds it as
                # "1." (granted B1/B2) or "1 ." (applications A1)
                text = re.sub(r"^\d+\s*\.\s*", "", text)
                claim_texts.append(f"### Claim {num}\n{text}\n")

    # Build markdown
    lines = [f"# {title}\n", f"**Publication Number:** {pub_number}\n"]

    if abstract_text:
        lines.append("## Abstract\n")
        lines.append(f"{abstract_text}\n")

    if description_parts:
        lines.append("## Description\n")
        lines.extend(description_parts)
        lines.append("")

    if claim_texts:
        lines.append(f"## Claims\n")
        for ct in claim_texts:
            lines.append(ct)
        lines.append("")

    markdown = "\n".join(lines)
    cache_path.write_text(markdown)

    abstract_snippet = abstract_text[:150] + "..." if len(abstract_text) > 150 else abstract_text

    return PatentFetchResult(
        success=True,
        publication_number=pub_number,
        title=title,
        abstract_snippet=abstract_snippet,
        claim_count=len(claim_texts),
        file_path=str(cache_path.relative_to(workspace)),
    )


