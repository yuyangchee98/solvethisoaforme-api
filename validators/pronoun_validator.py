"""Validator for detecting pronouns in claims."""

from validators.base import BaseValidator, ValidatorContext
from models.validation import AntecedentError


class PronounValidator(BaseValidator):
    """Detects pronouns which are not allowed in patent claims."""

    ERROR_PRONOUNS = {"it", "its", "they", "their", "them"}

    @property
    def name(self) -> str:
        return "pronoun_check"

    @property
    def description(self) -> str:
        return "Detect pronouns (it, its, they, etc.) which are not allowed in patent claims"

    def validate(self, context: ValidatorContext) -> list[AntecedentError]:
        """Find pronouns in the claim text."""
        errors = []

        for token in context.doc:
            if token.text.lower() in self.ERROR_PRONOUNS and token.pos_ == "PRON":
                errors.append(AntecedentError(
                    text=token.text,
                    np=token.text.lower(),
                    start=token.idx,
                    end=token.idx + len(token.text),
                    reason=f"Pronoun '{token.text}' not allowed in patent claims",
                ))

        return errors
