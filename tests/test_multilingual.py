from brag.multilingual import EUROPEAN_RULES, normalize_text


def test_nfc_composes_accents():
    assert normalize_text("Müller") == "Müller"


def test_strips_invisible_characters():
    assert normalize_text("Arbeits­markt​") == "Arbeitsmarkt"


def test_no_break_spaces_become_spaces():
    assert normalize_text("Vraiment ? Oui !") == "Vraiment ? Oui !"


def test_greek_question_mark_folds_to_semicolon():
    assert normalize_text("Τι κάνεις;") == "Τι κάνεις;"


def test_clause_level_keeps_elisions_and_hyphens():
    clause_delimiters = EUROPEAN_RULES.levels[2].delimiters
    assert "'" not in clause_delimiters and "-" not in clause_delimiters


def test_sentence_level_covers_closing_quotes():
    sentence_delimiters = EUROPEAN_RULES.levels[1].delimiters
    assert '." ' in sentence_delimiters and ".» " in sentence_delimiters and "։ " in sentence_delimiters
