"""Shared text-cleaning helpers used by the filtering and export stages."""
import re

MARKER_LABELS = {
    "brand story", "from the manufacturer", "from the brand",
    "product description", "a+ content",
}
STYLE_TAG_RE = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
CSS_BLOCK_RE = re.compile(r"(?:#|\.)[a-zA-Z0-9_-]+\s*\{[^}]*\}", re.DOTALL | re.IGNORECASE)

IGNORED_DETAILS_KEYS = {
    "package dimensions", "item model number", "asin",
    "date first available", "manufacturer", "upc",
}


def clean_element(s):
    if not isinstance(s, str):
        return ""
    original = s.strip()
    if not original or original.lower() in MARKER_LABELS:
        return ""
    cleaned = STYLE_TAG_RE.sub(" ", original)
    cleaned = CSS_BLOCK_RE.sub(" ", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def split_field_elements(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def clean_field(value):
    """Return (joined_clean_length, has_clean_content_bool)."""
    cleaned_elements = [clean_element(e) for e in split_field_elements(value)]
    cleaned_elements = [c for c in cleaned_elements if c]
    total_len = sum(len(c) for c in cleaned_elements)
    return total_len, len(cleaned_elements) > 0


def clean_length_of(value):
    return sum(len(clean_element(e)) for e in split_field_elements(value))


def sanitized_list(value):
    return [c for c in (clean_element(e) for e in split_field_elements(value)) if c]


def meaningful_details_length(value):
    if not value or not isinstance(value, dict):
        return 0
    valid_str = " ".join(
        f"{k}: {v}" for k, v in value.items()
        if str(k).strip().lower() not in IGNORED_DETAILS_KEYS and v
    )
    return len(valid_str)


def descriptive_length(record):
    title = record.get("title", "")
    title_str = title.strip() if isinstance(title, str) else ""
    return len(title_str) + clean_length_of(record.get("features")) + clean_length_of(record.get("description"))


def get_category_path(record):
    cats = record.get("categories", [])
    if isinstance(cats, list) and len(cats) > 0:
        return " > ".join(cats) if isinstance(cats[0], str) else str(cats)
    return "MISSING"
