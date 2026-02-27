"""Document processor for PDF files using pymupdf."""

import logging
from pathlib import Path

import pymupdf

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

        doc.close()

        # Detect scanned/image-only PDFs
        avg_chars = total_chars / page_count
        if avg_chars < _MIN_CHARS_PER_PAGE:
            return ProcessorResult(
                error=(
                    f"This PDF appears to be scanned/image-based "
                    f"({int(avg_chars)} chars/page average across {page_count} pages). "
                    f"Text extraction produced little usable text. "
                    f"Please paste the text content instead, or provide a text-based PDF."
                ),
            )

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

        return ProcessorResult(
            extracted_text=extracted_text,
            extracted_path=out_path,
            metadata={"pages": page_count, "total_chars": total_chars},
        )
