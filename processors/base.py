"""Base class and result type for document processors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProcessorResult:
    """Result of processing a document."""

    extracted_text: str = ""
    extracted_path: Path | None = None
    metadata: dict = field(default_factory=dict)
    error: str | None = None


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors."""

    @property
    @abstractmethod
    def supported_media_types(self) -> set[str]:
        """MIME types this processor handles."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """File extensions (without dot) this processor handles."""
        ...

    def can_process(self, media_type: str | None = None, filename: str | None = None) -> bool:
        """Check if this processor can handle the given file."""
        if media_type and media_type in self.supported_media_types:
            return True
        if filename:
            ext = Path(filename).suffix.lstrip(".").lower()
            if ext in self.supported_extensions:
                return True
        return False

    @abstractmethod
    def process(self, file_path: Path, output_dir: Path) -> ProcessorResult:
        """Process a document and extract text.

        Args:
            file_path: Path to the source document
            output_dir: Directory to write extracted output

        Returns:
            ProcessorResult with extracted text and metadata
        """
        ...
