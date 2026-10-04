"""spaCy NLP model singletons, loaded lazily on first use.

Loading en_core_web_trf pulls in torch and takes tens of seconds; deferring
it keeps container cold starts fast for endpoints that don't need NLP.
"""

import threading

import spacy

_lock = threading.Lock()
_nlp = None
_nlp_vectors = None


def get_nlp():
    """Transformer model (most accurate for patent text)."""
    global _nlp
    if _nlp is None:
        with _lock:
            if _nlp is None:
                _nlp = spacy.load("en_core_web_trf")
    return _nlp


def get_nlp_vectors():
    """Medium model with word vectors, used for similarity suggestions."""
    global _nlp_vectors
    if _nlp_vectors is None:
        with _lock:
            if _nlp_vectors is None:
                _nlp_vectors = spacy.load("en_core_web_md")
    return _nlp_vectors
