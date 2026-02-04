"""Validator for detecting re-introduction of inherited terms."""

from validators.base import BaseValidator, ValidatorContext
from models.validation import AntecedentError


class ReintroductionValidator(BaseValidator):
    """Checks for re-introduction of terms already defined in parent claims.

    Detects when a dependent claim introduces a term (with "a/an", "one or more", etc.)
    that was already introduced in a parent claim. This creates ambiguity about whether
    the term refers to the same element or a new one.

    Example:
        Claim 1: "analyzing data to obtain one or more pieces of material information"
        Claim 2 (depends on 1): "selecting one or more pieces of material information"

        This is ambiguous - should claim 2 use "the one or more pieces of material
        information" to clearly reference the element from claim 1?
    """

    @property
    def name(self) -> str:
        return "reintroduction_check"

    @property
    def description(self) -> str:
        return "Detect terms re-introduced in dependent claims that were already defined in parent claims"

    @property
    def default_enabled(self) -> bool:
        # Opt-in validator (disabled by default)
        return False

    def validate(self, context: ValidatorContext) -> list[AntecedentError]:
        """Check for re-introduction of inherited terms.

        Algorithm:
        1. Get all terms inherited from parent claims
        2. Find all terms introduced in the current claim
        3. Flag introductions that match inherited terms (should use "the" instead)

        Args:
            context: Validation context

        Returns:
            List of AntecedentError objects for re-introduction issues
        """
        errors = []

        # Skip independent claims (no inherited terms to conflict with)
        if not context.inherited_terms:
            return errors

        # Normalize inherited terms for comparison
        inherited_normalized = {term.lower() for term in context.inherited_terms}

        # Check each introduction in this claim
        for np in context.noun_phrases:
            if np.type == "introduction":
                # Check if this "new" term already exists in parent claims
                if np.np.lower() in inherited_normalized:
                    errors.append(AntecedentError(
                        text=np.text,
                        np=np.np,
                        start=np.start,
                        end=np.end,
                        reason=f"'{np.np}' re-introduced but already defined in parent claim; consider using 'the {np.np}' to reference existing element",
                        suggestion=f"the {np.np}",
                    ))

        return errors
