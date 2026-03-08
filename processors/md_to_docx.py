"""Convert Markdown text to a DOCX document in memory."""

import re
from io import BytesIO

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


def _apply_inline_formatting(paragraph, text: str):
    """Parse bold/italic markers and add formatted runs to a paragraph."""
    # Pattern matches **bold**, <u>underline</u>, ~~strikethrough~~, *italic*, or `code` segments
    pattern = re.compile(r"(\*\*(.+?)\*\*|<u>(.+?)</u>|~~(.+?)~~|\*(.+?)\*|`(.+?)`)")
    last_end = 0
    for m in pattern.finditer(text):
        # Add any plain text before this match
        if m.start() > last_end:
            run = paragraph.add_run(text[last_end : m.start()])
            run.font.size = Pt(11)
            run.font.name = "Calibri"
        if m.group(2):  # bold
            run = paragraph.add_run(m.group(2))
            run.bold = True
            run.font.size = Pt(11)
            run.font.name = "Calibri"
        elif m.group(3):  # underline
            run = paragraph.add_run(m.group(3))
            run.underline = True
            run.font.size = Pt(11)
            run.font.name = "Calibri"
        elif m.group(4):  # strikethrough
            run = paragraph.add_run(m.group(4))
            run.font.strike = True
            run.font.size = Pt(11)
            run.font.name = "Calibri"
        elif m.group(5):  # italic
            run = paragraph.add_run(m.group(5))
            run.italic = True
            run.font.size = Pt(11)
            run.font.name = "Calibri"
        elif m.group(6):  # inline code
            run = paragraph.add_run(m.group(6))
            run.font.name = "Courier New"
            run.font.size = Pt(10)
        last_end = m.end()
    # Trailing plain text
    if last_end < len(text):
        run = paragraph.add_run(text[last_end:])
        run.font.size = Pt(11)
        run.font.name = "Calibri"


def _strip_emsp_indent(text: str) -> tuple[str, int]:
    """Count and strip leading &emsp; entities, returning (stripped_text, indent_level)."""
    level = 0
    while text.startswith("&emsp;"):
        text = text[6:]  # len("&emsp;") == 6
        level += 1
    return text, level


def markdown_to_docx(markdown_text: str) -> BytesIO:
    """Convert markdown text to a DOCX file returned as an in-memory BytesIO.

    Supports headings (#-###), **bold**, <u>underline</u>, ~~strikethrough~~, *italic*, `inline code`,
    bullet lists (- item), numbered lists (1. item), fenced code blocks,
    &emsp; indentation, and plain paragraphs.
    """
    doc = Document()

    # Set default font
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    lines = markdown_text.split("\n")
    i = 0
    in_code_block = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]

        # Fenced code block toggle
        if line.strip().startswith("```"):
            if in_code_block:
                # End code block — flush accumulated lines
                p = doc.add_paragraph()
                run = p.add_run("\n".join(code_lines))
                run.font.name = "Courier New"
                run.font.size = Pt(9)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Blank line → skip
        if not stripped:
            i += 1
            continue

        # Strip &emsp; indentation before matching patterns
        stripped, indent_level = _strip_emsp_indent(stripped)

        # Headings
        heading_match = re.match(r"^(#{1,3})\s+(.*)", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            heading = doc.add_heading(level=level)
            _apply_inline_formatting(heading, text)
            i += 1
            continue

        # Bullet list item
        bullet_match = re.match(r"^[-*]\s+(.*)", stripped)
        if bullet_match:
            text = bullet_match.group(1)
            p = doc.add_paragraph(style="List Bullet")
            _apply_inline_formatting(p, text)
            i += 1
            continue

        # Numbered list item
        num_match = re.match(r"^\d+\.\s+(.*)", stripped)
        if num_match:
            text = num_match.group(1)
            p = doc.add_paragraph(style="List Number")
            _apply_inline_formatting(p, text)
            i += 1
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        if indent_level > 0:
            p.paragraph_format.left_indent = Inches(0.5 * indent_level)
        _apply_inline_formatting(p, stripped)
        i += 1

    # Flush any unclosed code block
    if code_lines:
        p = doc.add_paragraph()
        run = p.add_run("\n".join(code_lines))
        run.font.name = "Courier New"
        run.font.size = Pt(9)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
