"""spaCy NLP model singletons."""

import spacy

# Load spaCy transformer model (most accurate for patent text)
nlp = spacy.load("en_core_web_trf")

# Load medium model for word vector similarity (suggestions)
nlp_vectors = spacy.load("en_core_web_md")
