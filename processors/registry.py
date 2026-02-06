"""Registry for document processors."""

import logging
from pathlib import Path

from .base import BaseDocumentProcessor, ProcessorResult

logger = logging.getLogger(__name__)


class ProcessorRegistry:
    """Registry for document processors."""

    def __init__(self):
        self._processors: list[BaseDocumentProcessor] = []

    def register(self, processor: BaseDocumentProcessor) -> None:
        """Register a document processor."""
        self._processors.append(processor)

    def find_processor(
        self, media_type: str | None = None, filename: str | None = None
    ) -> BaseDocumentProcessor | None:
        """Find a processor that can handle the given file."""
        for proc in self._processors:
            if proc.can_process(media_type, filename):
                return proc
        return None

    def process_if_needed(
        self, file_path: Path, media_type: str | None, output_dir: Path
    ) -> ProcessorResult | None:
        """Process a file if a matching processor exists.

        Returns None if no processor matches (file handled as-is).
        """
        proc = self.find_processor(media_type, file_path.name)
        if proc is None:
            return None
        try:
            return proc.process(file_path, output_dir)
        except Exception as e:
            logger.warning("Processor %s failed for %s: %s", type(proc).__name__, file_path, e)
            return ProcessorResult(error=str(e))


# Singleton
_registry = ProcessorRegistry()


def get_processor_registry() -> ProcessorRegistry:
    """Get the global processor registry."""
    return _registry
