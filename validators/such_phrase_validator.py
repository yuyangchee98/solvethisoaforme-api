"""Validator for detecting 'such X' phrases."""

from validators.base import BaseValidator, ValidatorContext
from models.validation import AntecedentError


class SuchPhraseValidator(BaseValidator):
    """Detects 'such X' phrases which are invalid references."""

    @property
    def name(self) -> str:
        return "such_phrase_check"

    @property
    def description(self) -> str:
        return "Detect 'such X' phrases which are invalid antecedent references"

    def validate(self, context: ValidatorContext) -> list[AntecedentError]:
        """Find 'such' phrases in noun phrases."""
        errors = []

        for np in context.noun_phrases:
            if np.type == "error":  # Already classified as error by determiner check
                errors.append(AntecedentError(
                    text=np.text,
                    np=np.np,
                    start=np.start,
                    end=np.end,
                    reason=f"'such {np.np}' is not a valid reference",
                ))

        return errors
