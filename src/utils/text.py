import html
import re


def strip_html(text: str) -> str:
    """Strip HTML tags and decode entities to plain text."""
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
