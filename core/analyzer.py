"""Core claim analyzer orchestrating the validation pipeline."""

from core.nlp_models import nlp
from core.registry import get_registry
from models.requests import ParsedClaim
from models.responses import ClaimAnalysis
from models.validation import NounPhrase
from extractors import extract_noun_phrases, collect_inherited_terms
from validators.base import ValidatorContext


class ClaimAnalyzer:
    """Orchestrates claim analysis using pluggable validators."""

    def __init__(self):
        self.registry = get_registry()

    def analyze_claims(
        self,
        claims: list[ParsedClaim],
        enabled_validators: set[str] | None = None,
    ) -> list[ClaimAnalysis]:
        """Analyze all claims for antecedent basis errors.

        Args:
            claims: List of claims to analyze
            enabled_validators: Optional set of validator names to enable

        Returns:
            List of ClaimAnalysis objects
        """
        # Build claims map
        claims_map = {c.number: c for c in claims}
        intro_cache: dict[int, set[str]] = {}

        # Get enabled validators
        validators = self.registry.get_enabled_validators(enabled_validators)

        analyses = []

        for claim in claims:
            doc = nlp(claim.text)
            nps = extract_noun_phrases(doc)

            # Get inherited terms from ancestors
            inherited = collect_inherited_terms(claim.number, claims_map, intro_cache)

            # Separate introductions and references
            # Include both "introduction" (a/an) and "bare" (no determiner) as introductions
            introductions = [np for np in nps if np.type in ("introduction", "bare")]
            references = [np for np in nps if np.type == "reference"]

            # Create validation context
            context = ValidatorContext(
                doc=doc,
                noun_phrases=nps,
                inherited_terms=inherited,
                claim_text=claim.text,
                claim_number=claim.number,
            )

            # Run all enabled validators
            all_errors = []
            for validator in validators:
                errors = validator.validate(context)
                all_errors.extend(errors)

            analyses.append(ClaimAnalysis(
                claim_number=claim.number,
                claim_text=claim.text,
                introductions=introductions,
                references=references,
                inherited_terms=sorted(list(inherited)),
                antecedent_errors=all_errors,
            ))

        return analyses


def register_default_validators() -> None:
    """Register all default validators with the registry."""
    from validators import (
        PronounValidator,
        SuchPhraseValidator,
        AntecedentValidator,
        PrepositionalScopeValidator,
    )

    registry = get_registry()
    registry.register(PronounValidator())
    registry.register(SuchPhraseValidator())
    registry.register(AntecedentValidator())
    registry.register(PrepositionalScopeValidator())
