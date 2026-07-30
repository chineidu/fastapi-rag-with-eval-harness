from src.utils.text import strip_html


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
