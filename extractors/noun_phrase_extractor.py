"""Extract noun phrases from claim text."""

import spacy
from models.validation import NounPhrase


def extract_noun_phrases(doc: spacy.tokens.Doc) -> list[NounPhrase]:
    """Extract all noun phrases with their determiners.

    Args:
        doc: spaCy Doc object

    Returns:
        List of NounPhrase objects with type classification
    """
    results = []

    for chunk in doc.noun_chunks:
        # Find the determiner
        determiner = None
        det_token = None
        for token in chunk:
            if token.dep_ == "det":
                determiner = token.text.lower()
                det_token = token
                break

        # Get NP without determiner
        if det_token:
            np_start = det_token.idx + len(det_token.text_with_ws) - chunk.start_char
            np_text = chunk.text[np_start:].strip()
        else:
            np_text = chunk.text.strip()

        # Determine type
        if determiner in ("a", "an"):
            np_type = "introduction"
        elif determiner in ("the", "said"):
            np_type = "reference"
        elif determiner in ("such",):
            np_type = "error"
        elif determiner is None:
            np_type = "bare"
        else:
            np_type = "other"

        results.append(NounPhrase(
            text=chunk.text,
            np=np_text.lower(),
            determiner=determiner,
            start=chunk.start_char,
            end=chunk.end_char,
            type=np_type,
        ))

    return results
