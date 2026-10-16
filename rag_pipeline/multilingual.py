"""Language-agnostic text handling for European languages.

No per-document language detection is needed: Latin, Greek, Cyrillic,
Armenian and Georgian scripts all separate words with spaces and end
sentences with a small set of marks, so one rule set covers them.
"""

import re
import unicodedata

from chonkie.types import RecursiveLevel, RecursiveRules

# Invisible characters common in Office text: soft hyphen (German/Dutch
# hyphenation hints), zero-width space/joiners, BOM, word joiner.
_INVISIBLE = re.compile("[­​‌‍⁠﻿]")
# No-break spaces (French puts narrow ones before ? ! : ; and inside « »).
_NBSP = re.compile("[   ]")


def normalize_text(text: str) -> str:
    """NFC-normalise and strip invisible characters so identical words are
    identical strings (for embeddings, entity names and token counts).

    NFC also folds the Greek question mark (U+037E) into ";".
    """
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    return _NBSP.sub(" ", text)


# Closing quotes used across Europe: " ” » « „ ’ (German closes with “, Danish/Swiss with »/«).
_CLOSERS = ['"', "”", "“", "»", "«", "’"]
_SENTENCE_ENDS = [".", "!", "?", "…", ";", "։"]  # … ; (Greek question mark) ։ (Armenian full stop)

EUROPEAN_RULES = RecursiveRules(levels=[
    # 1. Paragraphs / lines
    RecursiveLevel(delimiters=["\n\n", "\r\n", "\n", "\r"]),
    # 2. Sentences, including ones that end inside closing quotes (`..." `, `...» `)
    RecursiveLevel(delimiters=[e + c + " " for e in _SENTENCE_ENDS for c in _CLOSERS] + [e + " " for e in _SENTENCE_ENDS]),
    # 3. Clauses. Deliberately excludes ' and - (Chonkie's default level does
    # split on them), which would break French/Italian elisions (l'homme,
    # dell'anno) and hyphenated compounds (Arbeits-, e-mail, Jean-Luc).
    RecursiveLevel(delimiters=[", ", ": ", " – ", " — ", " (", ") "]),
    # 4. Words, then 5. characters as a last resort
    RecursiveLevel(whitespace=True),
    RecursiveLevel(),
])
