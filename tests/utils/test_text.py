from src.utils.text import clean_query_text, strip_html


class TestStripHtml:
    def test_strips_simple_tags(self) -> None:
        assert strip_html("<p>Hello</p>") == "Hello"

    def test_strips_nested_tags(self) -> None:
        assert strip_html("<div><p>Hello <b>world</b></p></div>") == "Hello world"

    def test_decodes_html_entities(self) -> None:
        assert (
            strip_html("a &amp; b &lt; c &gt; d &quot; e &#39; f")
            == "a & b < c > d \" e ' f"
        )

    def test_handles_empty_string(self) -> None:
        assert strip_html("") == ""

    def test_handles_blank_html(self) -> None:
        assert strip_html("<br><hr>") == ""

    def test_preserves_non_html_text(self) -> None:
        assert strip_html("plain text no tags") == "plain text no tags"

    def test_handles_code_blocks(self) -> None:
        assert strip_html("<code>print('hello')</code>") == "print('hello')"

    def test_strips_attributes(self) -> None:
        assert (
            strip_html('<a href="http://example.com" target="_blank">link</a>')
            == "link"
        )

    def test_strips_self_closing_tags(self) -> None:
        assert strip_html("text<br/>more<img src='x'/>end") == "textmoreend"

    def test_strips_whitespace_after(self) -> None:
        result = strip_html("<p>  hello world  </p>")
        assert result == "hello world"


_GITHUB_TEMPLATE_BODY = """\
### First Check

- [X] I added a very descriptive title here.
- [X] I used the GitHub search to find a similar question and didn't find it.

### Commit to Help

- [X] I commit to help with one of those options

### Example Code

```python
from fastapi import FastAPI, Form
from pydantic import BaseModel

class FormData(BaseModel):
    username: str
```

### Description

I was excited to see this new feature, but I am getting error messages when
I try to use it.

### Operating System

Windows

### FastAPI Version

0.114.0

### Pydantic Version

2.9.0

### Python Version

3.11.8

### Additional Context

_No response_
"""


class TestCleanQueryTextGithub:
    def test_keeps_description_and_drops_boilerplate(self) -> None:
        result = clean_query_text(
            _GITHUB_TEMPLATE_BODY, "github", "JSON is not parsed natively"
        )
        assert result.startswith("JSON is not parsed natively\n\n")
        assert "I was excited to see this new feature" in result
        assert "First Check" not in result
        assert "I commit to help" not in result
        assert "Operating System" not in result
        assert "0.114.0" not in result

    def test_keeps_example_code_as_context(self) -> None:
        result = clean_query_text(_GITHUB_TEMPLATE_BODY, "github", "title")
        assert "Example code:" in result
        assert "class FormData(BaseModel):" in result

    def test_handles_crlf_line_endings(self) -> None:
        crlf_body = _GITHUB_TEMPLATE_BODY.replace("\n", "\r\n")
        result = clean_query_text(crlf_body, "github", "title")
        assert "I was excited to see this new feature" in result
        assert "First Check" not in result

    def test_freeform_body_without_sections_is_kept(self) -> None:
        body = "Hi there,\n\nI have found synchronous dependencies easy to deal with."
        result = clean_query_text(body, "github", "Async dependencies?")
        assert result.startswith("Async dependencies?\n\n")
        assert "synchronous dependencies" in result

    def test_bold_description_headers_are_kept_as_freeform(self) -> None:
        body = "**Description**\n\nHow can I use peewee ORM with fastapi?"
        result = clean_query_text(body, "github", "Peewee usage")
        assert "How can I use peewee ORM with fastapi?" in result
        assert "Peewee usage" in result

    def test_empty_body_returns_title(self) -> None:
        assert clean_query_text("", "github", "Just a title") == "Just a title"

    def test_drops_other_template_headers_when_no_description(self) -> None:
        body = "### Privileged issue\n\n- [x] I'm @tiangolo\n\n### Issue Content\n\n## Description\nFixed a typo."
        result = clean_query_text(body, "github", "docs: Fix typo")
        assert "Fixed a typo." in result
        assert "Privileged issue" not in result
        assert "tiangolo" not in result


class TestCleanQueryTextStackOverflow:
    def test_strips_html_tags(self) -> None:
        body = "<p>I tried to run <code>uvicorn api:app</code> but it fails.</p>"
        result = clean_query_text(body, "stackoverflow", "any title")
        assert result == "I tried to run uvicorn api:app but it fails."

    def test_preserves_markdown_plain_text(self) -> None:
        body = "How do I set a default value for a query parameter?\n\n```python\nq: str = None\n```"
        result = clean_query_text(body, "stackoverflow", "any title")
        assert result == body

    def test_does_not_prepend_title(self) -> None:
        result = clean_query_text("Plain question text.", "stackoverflow", "Title")
        assert result == "Plain question text."

    def test_handles_empty_body(self) -> None:
        assert clean_query_text("", "stackoverflow", "title") == ""
