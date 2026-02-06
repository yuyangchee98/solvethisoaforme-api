"""Document processors for extracting text from uploaded files."""

from .base import BaseDocumentProcessor, ProcessorResult
from .registry import ProcessorRegistry, get_processor_registry
from .docx_processor import DocxProcessor

# Auto-register processors
get_processor_registry().register(DocxProcessor())

__all__ = [
    "BaseDocumentProcessor",
    "ProcessorResult",
    "ProcessorRegistry",
    "get_processor_registry",
    "DocxProcessor",
]
