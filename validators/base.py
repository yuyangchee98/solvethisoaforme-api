"""Base classes and protocols for validators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import spacy
from models.validation import NounPhrase, AntecedentError


@dataclass
class ValidatorContext:
    """Context passed to validators."""
    doc: spacy.tokens.Doc
    noun_phrases: list[NounPhrase]
    inherited_terms: set[str]
    claim_text: str
    claim_number: int


class BaseValidator(ABC):
    """Base class for all antecedent validators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this validator."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this validator checks."""
        pass

    @property
    def default_enabled(self) -> bool:
        """Whether this validator is enabled by default.

        Override to return False for opt-in validators.
        """
        return True

    @abstractmethod
    def validate(self, context: ValidatorContext) -> list[AntecedentError]:
        """Run validation and return any errors found.

        Args:
            context: Validation context containing claim data

        Returns:
            List of AntecedentError objects (empty if no errors)
        """
        pass
