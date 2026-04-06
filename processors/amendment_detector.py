"""Detect and classify strikethrough/underline text in scanned patent amendment PDFs.

Uses horizontal line detection (numpy) + Google Vision API (word bounding boxes)
to identify which words are struck through (deleted) vs underlined (added).

Only triggered for scanned PDFs that contain amendment markers like "(Currently Amended)".
"""

import base64
import logging
import os
import time
from dataclasses import dataclass

import httpx
import numpy as np
import pymupdf

logger = logging.getLogger(__name__)

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
_MIN_HORIZONTAL_RUN = 80  # minimum pixel run length at 300dpi to count as a line
_OVERLAP_THRESHOLD = 0.5  # line must cover >50% of word width


@dataclass
class ClassifiedWord:
    text: str
    formatting: str  # "normal", "strikethrough", "underline"
    x_min: int
    y_min: int
    x_max: int
    y_max: int


def has_amendment_markers(ocr_text: str) -> bool:
    """Check if OCR text suggests this is a patent amendment with markup."""
    lower = ocr_text.lower()
    return "currently amended" in lower


def _detect_horizontal_lines(page: pymupdf.Page) -> list[tuple[int, int, int, int]]:
    """Detect horizontal line groups in a page image.

    Returns list of (y_min, y_max, x_min, x_max) in 300dpi pixel coords.
    """
    pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    binary = (arr < 128).astype(np.uint8)

    raw: list[tuple[int, int, int]] = []
    for y in range(arr.shape[0]):
        row = binary[y]
        in_run = False
        run_start = 0
        for x in range(len(row)):
            if row[x] == 1 and not in_run:
                in_run = True
                run_start = x
            elif row[x] == 0 and in_run:
                in_run = False
                if x - run_start >= _MIN_HORIZONTAL_RUN:
                    raw.append((y, run_start, x))
        if in_run and len(row) - run_start >= _MIN_HORIZONTAL_RUN:
            raw.append((y, run_start, len(row)))

    if not raw:
        return []

    # Cluster adjacent rows into line groups
    raw.sort()
    groups: list[list[tuple[int, int, int]]] = []
    current = [raw[0]]
    for line in raw[1:]:
        prev = current[-1]
        if line[0] - prev[0] <= 4 and abs(line[1] - prev[1]) < 20:
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    groups.append(current)

    return [
        (
            min(l[0] for l in g),
            max(l[0] for l in g),
            min(l[1] for l in g),
            max(l[2] for l in g),
        )
        for g in groups
    ]


def _vision_ocr_words(page: pymupdf.Page, api_key: str) -> list[dict]:
    """Call Google Vision API on a page, return word bounding boxes.

    Each word: {"text": str, "x_min": int, "y_min": int, "x_max": int, "y_max": int}
    Coordinates are in 300dpi pixel space.
    """
    pix = page.get_pixmap(dpi=300)
    png_bytes = pix.tobytes("png")

    resp = httpx.post(
        VISION_URL,
        params={"key": api_key},
        json={
            "requests": [{
                "image": {"content": base64.b64encode(png_bytes).decode()},
                "features": [{"type": "TEXT_DETECTION"}],
            }]
        },
        timeout=60,
    )
    resp.raise_for_status()

    anns = resp.json().get("responses", [{}])[0].get("textAnnotations", [])
    words = []
    for a in anns[1:]:  # skip first entry (full text block)
        v = a["boundingPoly"]["vertices"]
        words.append({
            "text": a["description"],
            "x_min": min(p.get("x", 0) for p in v),
            "y_min": min(p.get("y", 0) for p in v),
            "x_max": max(p.get("x", 0) for p in v),
            "y_max": max(p.get("y", 0) for p in v),
        })
    return words


def _classify_words(
    pixel_lines: list[tuple[int, int, int, int]],
    vision_words: list[dict],
) -> list[ClassifiedWord]:
    """Classify each word as normal, strikethrough, or underline.

    Cross-references detected horizontal lines with word bounding boxes.
    A line through the middle of a word = strikethrough.
    A line at the bottom of a word = underline.
    If both are present, strikethrough wins.
    """
    classified = []
    for w in vision_words:
        wx0, wy0, wx1, wy1 = w["x_min"], w["y_min"], w["x_max"], w["y_max"]
        word_h = wy1 - wy0
        word_w = wx1 - wx0
        if word_h <= 0 or word_w <= 0:
            continue

        is_strike = False
        is_underline = False

        for ly_min, ly_max, lx_min, lx_max in pixel_lines:
            line_mid_y = (ly_min + ly_max) / 2

            # Require >50% horizontal overlap
            overlap = max(0, min(wx1, lx_max) - max(wx0, lx_min))
            if overlap / word_w < _OVERLAP_THRESHOLD:
                continue

            # Vertical proximity
            if not (wy0 - 15 <= line_mid_y <= wy1 + 15):
                continue

            # Position ratio: 0 = top of word, 1 = bottom
            line_ratio = (line_mid_y - wy0) / word_h
            if 0.25 <= line_ratio <= 0.65:
                is_strike = True
            elif 0.75 <= line_ratio <= 1.15:
                is_underline = True

        # Strikethrough wins if both present
        if is_strike:
            fmt = "strikethrough"
        elif is_underline:
            fmt = "underline"
        else:
            fmt = "normal"

        classified.append(ClassifiedWord(
            text=w["text"], formatting=fmt,
            x_min=wx0, y_min=wy0, x_max=wx1, y_max=wy1,
        ))

    return classified


def extract_amendment_text(
    doc: pymupdf.Document,
    ocr_pages: list[str] | None = None,
) -> str | None:
    """Extract text from a scanned amendment PDF with strikethrough handling.

    Only calls Vision API on pages that have horizontal lines (amendments).
    Pages without lines reuse the existing OCR text.

    Args:
        doc: Open pymupdf document.
        ocr_pages: Pre-extracted OCR text per page (from pymupdf OCR).
            Used for pages that don't need Vision processing.

    Returns markdown text with ~~strikethrough~~ markers, or None if
    Vision API is unavailable or the PDF doesn't contain amendments.
    """
    api_key = os.environ.get("GOOGLE_VISION_API_KEY", "")
    if not api_key:
        logger.info("GOOGLE_VISION_API_KEY not set, skipping amendment detection")
        return None

    t0 = time.monotonic()
    page_count = len(doc)
    parts: list[str] = []
    vision_calls = 0

    for pg_idx, page in enumerate(doc):
        # Line detection is free/local — only call Vision if lines found
        lines = _detect_horizontal_lines(page)

        if not lines:
            # No amendment lines — reuse existing OCR text, skip Vision API
            if ocr_pages and pg_idx < len(ocr_pages):
                page_text = ocr_pages[pg_idx]
            else:
                # Fallback: quick pymupdf OCR
                tp = page.get_textpage_ocr(language="eng", dpi=150, full=True)
                page_text = page.get_text("text", textpage=tp)
        else:
            # Amendment lines detected — call Vision API for precise word bboxes
            words = _vision_ocr_words(page, api_key)
            vision_calls += 1
            classified = _classify_words(lines, words)

            # Build text: wrap strikethrough in ~~
            tokens: list[str] = []
            for cw in classified:
                if cw.formatting == "strikethrough":
                    tokens.append(f"~~{cw.text}~~")
                else:
                    tokens.append(cw.text)
            page_text = " ".join(tokens)

        page_text = page_text.strip()
        if not page_text:
            continue
        if page_count > 1:
            parts.append(f"<!-- Page {pg_idx + 1} -->")
        parts.append(page_text)

    elapsed = time.monotonic() - t0
    logger.info(
        "Amendment-aware extraction completed in %.1fs (%d pages, %d Vision API calls)",
        elapsed, page_count, vision_calls,
    )

    return "\n\n".join(parts) + "\n"
