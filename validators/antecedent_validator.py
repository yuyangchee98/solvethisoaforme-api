"""Validator for checking antecedent basis of references."""

from validators.base import BaseValidator, ValidatorContext
from models.validation import AntecedentError
from utils.similarity import find_closest_match


class AntecedentValidator(BaseValidator):
    """Checks that 'the/said' references have valid antecedents."""

    @property
    def name(self) -> str:
        return "antecedent_check"

    @property
    def description(self) -> str:
        return "Check that 'the/said' references have previously introduced antecedents"

    def validate(self, context: ValidatorContext) -> list[AntecedentError]:
        """Check references against inherited terms."""
        errors = []

        # Get references (noun phrases with "the" or "said")
        references = [np for np in context.noun_phrases if np.type == "reference"]

        for ref in references:
            if ref.np not in context.inherited_terms:
                # Find closest match for suggestion
                suggestion, score = find_closest_match(ref.np, context.inherited_terms)
                errors.append(AntecedentError(
                    text=ref.text,
                    np=ref.np,
                    start=ref.start,
                    end=ref.end,
                    reason=f"No antecedent for '{ref.np}'",
                    suggestion=suggestion,
                    suggestion_score=round(score, 3) if score else None,
                ))

        return errors
