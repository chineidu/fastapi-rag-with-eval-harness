import html
import re

_GITHUB_BOILERPLATE_SECTIONS = frozenset(
    {
        "first check",
        "commit to help",
        "operating system",
        "operating system details",
        "fastapi version",
        "pydantic version",
        "python version",
        "additional context",
        "environment",
        "privileged issue",
    }
)
_CHECKBOX_PATTERN = re.compile(r"^-\s+\[[ xX]\]\s*.*$", flags=re.MULTILINE)
_SECTION_HEADER_PATTERN = re.compile(r"^###\s+(.+)$", flags=re.MULTILINE)


def strip_html(text: str) -> str:
    """Strip HTML tags and decode entities to plain text."""
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _normalize_line_endings(text: str) -> str:
    """Convert CRLF/CR line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a GitHub-style body into ``(header, content)`` sections.

    Content runs from a ``### header`` line up to the next ``### header``
    (or the end of the text). Content is stripped of surrounding whitespace
    and leading checkbox lines.
    """
    matches = list(_SECTION_HEADER_PATTERN.finditer(body))
    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        content = _CHECKBOX_PATTERN.sub("", content).strip()
        sections.append((match.group(1).strip().lower(), content))
    return sections


def _extract_section(sections: list[tuple[str, str]], header: str) -> str | None:
    """Return the first section content whose header matches ``header``."""
    for section_header, content in sections:
        if section_header == header:
            return content
    return None


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of 3+ newlines into a single blank line."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean_github_body(body: str, title: str) -> str:
    """Clean a GitHub discussion body into retrieval-ready question text."""
    body = _normalize_line_endings(body)
    sections = _split_sections(body)

    description = _extract_section(sections, "description")
    if description is not None:
        example_code = _extract_section(sections, "example code")
        parts = [title, description]
        if example_code:
            parts.append(f"Example code:\n{example_code}")
        return _collapse_whitespace("\n\n".join(parts))

    if not sections:
        body = _CHECKBOX_PATTERN.sub("", body)
        return _collapse_whitespace(f"{title}\n\n{body}")

    meaningful = [
        content
        for header, content in sections
        if header not in _GITHUB_BOILERPLATE_SECTIONS and content
    ]
    parts = [title, *meaningful]
    return _collapse_whitespace("\n\n".join(parts))


def clean_query_text(body: str, source: str, title: str = "") -> str:
    """Produce a retrieval-ready question from a raw discussion/QA body.

    GitHub discussion bodies carry issue-template boilerplate
    (``### First Check``, ``### Commit to Help``, etc.); this function keeps
    the ``### Description`` section (plus the ``### Example Code`` block as
    context) and drops the rest. Stack Overflow bodies are kept as-is with
    HTML tags stripped.

    Parameters
    ----------
    body : str
        Raw question body (GitHub discussion markdown or Stack Overflow
        text).
    source : str
        Origin of the record, ``"github"`` or ``"stackoverflow"``.
    title : str
        Human-readable title of the original question, prepended for GitHub
        records.

    Returns
    -------
    str
        Cleaned question text suitable for the ``query_text`` field of
        ground truth.
    """
    if source != "github":
        return strip_html(body)
    return _clean_github_body(body, title)
