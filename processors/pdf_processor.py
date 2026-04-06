"""Document processor for PDF files using pymupdf."""

import logging
import time
from pathlib import Path

import pymupdf

from .amendment_detector import extract_amendment_text, has_amendment_markers
from .base import BaseDocumentProcessor, ProcessorResult

logger = logging.getLogger(__name__)

# Minimum average characters per page to consider a PDF text-based (not scanned)
_MIN_CHARS_PER_PAGE = 100


class PdfProcessor(BaseDocumentProcessor):
    """Extracts text from PDF files as markdown."""

    @property
    def supported_media_types(self) -> set[str]:
        return {"application/pdf"}

    @property
    def supported_extensions(self) -> set[str]:
        return {"pdf"}

    def process(self, file_path: Path, output_dir: Path) -> ProcessorResult:
        try:
            doc = pymupdf.open(str(file_path))
        except Exception as e:
            return ProcessorResult(error=f"Failed to open PDF: {e}")

        page_count = len(doc)
        if page_count == 0:
            doc.close()
            return ProcessorResult(error="PDF has no pages.")

        pages: list[str] = []
        total_chars = 0

        for page in doc:
            text = page.get_text("text")
            total_chars += len(text)
            pages.append(text)

        # Detect scanned/image-only PDFs and attempt OCR
        avg_chars = total_chars / page_count
        used_ocr = False

        if avg_chars < _MIN_CHARS_PER_PAGE:
            logger.info(
                "PDF '%s' appears scanned (%d chars/page avg). Attempting OCR...",
                file_path.name,
                int(avg_chars),
            )
            try:
                t0 = time.monotonic()
                ocr_pages: list[str] = []
                ocr_total_chars = 0
                for page in doc:
                    tp = page.get_textpage_ocr(language="eng", dpi=150, full=True)
                    text = page.get_text("text", textpage=tp)
                    ocr_total_chars += len(text)
                    ocr_pages.append(text)
                elapsed = time.monotonic() - t0
                ocr_avg = ocr_total_chars / page_count

                logger.info(
                    "OCR completed for '%s' in %.1fs: %d chars/page avg (was %d)",
                    file_path.name,
                    elapsed,
                    int(ocr_avg),
                    int(avg_chars),
                )

                if ocr_avg >= _MIN_CHARS_PER_PAGE:
                    pages = ocr_pages
                    total_chars = ocr_total_chars
                    used_ocr = True

                    # Check if OCR text contains amendment markers —
                    # if so, re-extract with Vision API + line detection
                    # to properly handle strikethrough/underline formatting
                    joined_ocr = "\n".join(ocr_pages)
                    if has_amendment_markers(joined_ocr):
                        logger.info(
                            "Amendment markers found in '%s', attempting Vision-based extraction",
                            file_path.name,
                        )
                        try:
                            amendment_text = extract_amendment_text(doc, ocr_pages=ocr_pages)
                            if amendment_text:
                                doc.close()
                                out_path = output_dir / f"{file_path.stem}.extracted.md"
                                out_path.write_text(amendment_text, encoding="utf-8")
                                metadata = {
                                    "pages": page_count,
                                    "total_chars": len(amendment_text),
                                    "ocr": True,
                                    "amendment_aware": True,
                                }
                                return ProcessorResult(
                                    extracted_text=amendment_text,
                                    extracted_path=out_path,
                                    metadata=metadata,
                                )
                        except Exception as e:
                            logger.warning(
                                "Amendment-aware extraction failed for '%s': %s. "
                                "Falling back to plain OCR.",
                                file_path.name, e,
                            )
                else:
                    doc.close()
                    return ProcessorResult(
                        error=(
                            f"This PDF appears to be scanned/image-based "
                            f"({int(avg_chars)} chars/page average across {page_count} pages). "
                            f"OCR was attempted but produced little usable text "
                            f"({int(ocr_avg)} chars/page). "
                            f"Please paste the text content instead, or provide a text-based PDF."
                        ),
                    )
            except Exception as e:
                logger.warning("OCR failed for '%s': %s", file_path.name, e)
                doc.close()
                return ProcessorResult(
                    error=(
                        f"This PDF appears to be scanned/image-based "
                        f"({int(avg_chars)} chars/page average across {page_count} pages). "
                        f"Text extraction produced little usable text. "
                        f"OCR was attempted but failed ({e}). "
                        f"Ensure Tesseract is installed (apt-get install tesseract-ocr). "
                        f"Or paste the text content instead."
                    ),
                )

        doc.close()

        # Build extracted text with page markers
        parts: list[str] = []
        for i, text in enumerate(pages, 1):
            # Normalize whitespace: collapse runs of 3+ newlines to 2
            cleaned = text.strip()
            if not cleaned:
                continue
            # Add page separator for multi-page docs
            if page_count > 1:
                parts.append(f"<!-- Page {i} -->")
            parts.append(cleaned)

        extracted_text = "\n\n".join(parts) + "\n"

        out_path = output_dir / f"{file_path.stem}.extracted.md"
        out_path.write_text(extracted_text, encoding="utf-8")

        metadata = {"pages": page_count, "total_chars": total_chars}
        if used_ocr:
            metadata["ocr"] = True

        return ProcessorResult(
            extracted_text=extracted_text,
            extracted_path=out_path,
            metadata=metadata,
        )
