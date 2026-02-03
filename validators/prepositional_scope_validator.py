"""Validator for checking prepositional scope consistency."""

import spacy
from validators.base import BaseValidator, ValidatorContext
from models.validation import AntecedentError


def get_prepositional_objects(doc: spacy.tokens.Doc) -> set[str]:
    """Extract terms that appear as prepositional objects.

    Returns normalized term strings that are objects of prepositions.
    Example: "corpus of document portions" → returns "document portions"

    Args:
        doc: spaCy Doc object

    Returns:
        Set of terms that appear as prepositional objects
    """
    pobj_terms = set()

    for token in doc:
        if token.dep_ == "pobj":
            # Get the full noun phrase starting from this pobj
            # Handle compounds like "document portions"
            phrase_tokens = [token]

            # Look for compound modifiers (words that modify this noun)
            for child in token.children:
                if child.dep_ == "compound":
                    phrase_tokens.insert(0, child)

            # Build the phrase
            phrase = " ".join(t.text for t in phrase_tokens).lower()
            pobj_terms.add(phrase)

    return pobj_terms


class PrepositionalScopeValidator(BaseValidator):
    """Checks for prepositional scope mismatches.

    Detects when a term is referenced with "the" but was only introduced
    as the object of a preposition (e.g., "corpus of document portions"
    cannot later be referenced as "the document portions" unless "document portions"
    was independently introduced).
    """

    @property
    def name(self) -> str:
        return "prepositional_scope"

    @property
    def description(self) -> str:
        return "Detect terms referenced with 'the' that were only introduced as prepositional objects"

    @property
    def default_enabled(self) -> bool:
        # Opt-in validator (disabled by default)
        return False

    def validate(self, context: ValidatorContext) -> list[AntecedentError]:
        """Check for prepositional scope issues.

        Algorithm:
        1. Find all terms that appear as prepositional objects (pobj)
        2. Find all terms that were directly introduced (not just as pobj)
        3. Flag references to terms that were ONLY introduced as pobj

        Args:
            context: Validation context

        Returns:
            List of AntecedentError objects for scope violations
        """
        errors = []

        # Step 1: Get all terms that appear as prepositional objects
        pobj_terms = get_prepositional_objects(context.doc)

        # Step 2: Get terms that were directly introduced (not as pobj)
        directly_introduced = set()
        for np in context.noun_phrases:
            if np.type in ("introduction", "bare"):
                # Check if this NP's root token is NOT a prepositional object
                np_span = context.doc.char_span(np.start, np.end)
                if np_span:
                    # Get the head/root of this noun phrase
                    root = np_span.root
                    # If the root is not a pobj, this was a direct introduction
                    if root.dep_ != "pobj":
                        directly_introduced.add(np.np)

        # Step 3: Check references against prepositional-only terms
        for np in context.noun_phrases:
            if np.type == "reference":  # "the X" or "said X"
                # Is this term only introduced as pobj in this claim?
                if np.np in pobj_terms and np.np not in directly_introduced:
                    # Flag as error - the term was only introduced as a prepositional object
                    # Note: This may flag cases where parent claims introduced the term properly
                    # but the current claim re-introduces it as pobj. This encourages cleaner
                    # claim drafting (don't re-introduce parent terms in prepositional phrases)
                    errors.append(AntecedentError(
                        text=np.text,
                        np=np.np,
                        start=np.start,
                        end=np.end,
                        reason=f"'{np.np}' only appears as object of preposition; not independently introduced",
                        suggestion=None,
                    ))

        return errors
