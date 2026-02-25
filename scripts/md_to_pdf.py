"""Convert PRESENTATION_OUTLINE.md to a clean PDF using reportlab."""

import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

MD_PATH = Path(__file__).parent.parent / "PRESENTATION_OUTLINE.md"
PDF_PATH = Path(__file__).parent.parent / "PRESENTATION_OUTLINE.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
DARK   = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#4f8ef7")
LIGHT  = colors.HexColor("#f0f4ff")
MUTED  = colors.HexColor("#6b7280")
WHITE  = colors.white

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

S = {
    "doc_title": ParagraphStyle(
        "doc_title", fontSize=22, leading=28, textColor=DARK,
        fontName="Helvetica-Bold", spaceAfter=6, alignment=TA_CENTER
    ),
    "slide_title": ParagraphStyle(
        "slide_title", fontSize=15, leading=20, textColor=ACCENT,
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4
    ),
    "slide_subtitle": ParagraphStyle(
        "slide_subtitle", fontSize=10, leading=14, textColor=MUTED,
        fontName="Helvetica-Oblique", spaceAfter=8
    ),
    "h3": ParagraphStyle(
        "h3", fontSize=11, leading=15, textColor=DARK,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3
    ),
    "body": ParagraphStyle(
        "body", fontSize=9, leading=14, textColor=DARK,
        fontName="Helvetica", spaceAfter=3
    ),
    "bullet": ParagraphStyle(
        "bullet", fontSize=9, leading=13, textColor=DARK,
        fontName="Helvetica", leftIndent=14, firstLineIndent=-10,
        spaceAfter=2, bulletIndent=4
    ),
    "bullet2": ParagraphStyle(
        "bullet2", fontSize=8.5, leading=12, textColor=DARK,
        fontName="Helvetica", leftIndent=26, firstLineIndent=-10,
        spaceAfter=1
    ),
    "code": ParagraphStyle(
        "code", fontSize=8, leading=11, textColor=colors.HexColor("#1e3a5f"),
        fontName="Courier", backColor=LIGHT,
        leftIndent=12, rightIndent=12, spaceBefore=4, spaceAfter=4
    ),
    "note": ParagraphStyle(
        "note", fontSize=8, leading=11, textColor=MUTED,
        fontName="Helvetica-Oblique", spaceAfter=4
    ),
}

TABLE_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0),  ACCENT),
    ("TEXTCOLOR",   (0, 0), (-1, 0),  WHITE),
    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, 0),  8),
    ("BACKGROUND",  (0, 1), (-1, -1), LIGHT),
    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",    (0, 1), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",  (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0,0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",(0, 0), (-1, -1), 6),
])


# ── Parser ────────────────────────────────────────────────────────────────────

def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_inline(text):
    """Bold, italic, code spans — code spans are protected first to avoid _ clashes."""
    # 1. Extract code spans and replace with placeholders
    placeholders = {}
    def save_code(m):
        key = f"\x00CODE{len(placeholders)}\x00"
        placeholders[key] = f'<font name="Courier" color="#1e3a5f">{escape(m.group(1))}</font>'
        return key
    text = re.sub(r"`([^`]+)`", save_code, text)

    # 2. Escape HTML in the remaining text
    text = escape(text)

    # 3. Bold / italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)

    # 4. Restore code spans (placeholders are not HTML-escaped)
    for key, val in placeholders.items():
        text = text.replace(escape(key), val)

    return text


def build_table(lines):
    """Parse a markdown pipe table into a reportlab Table."""
    rows = []
    for line in lines:
        if re.match(r"^\|[-| :]+\|$", line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return None
    page_w = A4[0] - 4 * cm
    col_w = page_w / len(rows[0])
    tbl = Table(rows, colWidths=[col_w] * len(rows[0]))
    tbl.setStyle(TABLE_STYLE)
    return tbl


def md_to_flowables(md_text):
    story = []
    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf = []
    table_buf = []

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # ── fenced code block ──
        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                block = "\n".join(code_buf)
                for cl in block.split("\n"):
                    story.append(Paragraph(escape(cl) or " ", S["code"]))
                story.append(Spacer(1, 4))
            i += 1
            continue
        if in_code:
            code_buf.append(raw)
            i += 1
            continue

        # ── pipe table ──
        if stripped.startswith("|"):
            table_buf.append(stripped)
            i += 1
            # collect all consecutive table lines
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_buf.append(lines[i].strip())
                i += 1
            tbl = build_table(table_buf)
            if tbl:
                story.append(Spacer(1, 6))
                story.append(tbl)
                story.append(Spacer(1, 8))
            table_buf = []
            continue

        # ── HR ──
        if re.match(r"^---+$", stripped):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#d1d5db")))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # ── H1 (doc title) ──
        if stripped.startswith("# ") and not stripped.startswith("## "):
            text = stripped[2:].strip()
            story.append(Spacer(1, 8))
            story.append(Paragraph(parse_inline(text), S["doc_title"]))
            story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT))
            story.append(Spacer(1, 10))
            i += 1
            continue

        # ── H2 (slide header) → page break + slide banner ──
        if stripped.startswith("## "):
            text = stripped[3:].strip()
            # page break before every slide except the very first
            if story:
                story.append(PageBreak())
            # slide number strip  e.g. "Slide 1 — Overview"
            story.append(Paragraph(parse_inline(text), S["slide_title"]))
            story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
            story.append(Spacer(1, 6))
            i += 1
            continue

        # ── H3 ──
        if stripped.startswith("### "):
            text = stripped[4:].strip()
            story.append(Paragraph(parse_inline(text), S["h3"]))
            i += 1
            continue

        # ── H4 ──
        if stripped.startswith("#### "):
            text = stripped[5:].strip()
            story.append(Paragraph(f"<b>{parse_inline(text)}</b>", S["body"]))
            i += 1
            continue

        # ── bullet level 1 ──
        if re.match(r"^[-*] ", stripped):
            text = stripped[2:].strip()
            story.append(Paragraph(f"• {parse_inline(text)}", S["bullet"]))
            i += 1
            continue

        # ── bullet level 2 (leading spaces) ──
        if re.match(r"^ {2,4}[-*] ", raw):
            text = re.sub(r"^ {2,4}[-*] ", "", raw).strip()
            story.append(Paragraph(f"◦ {parse_inline(text)}", S["bullet2"]))
            i += 1
            continue

        # ── numbered list ──
        if re.match(r"^\d+\. ", stripped):
            text = re.sub(r"^\d+\. ", "", stripped)
            num = re.match(r"^(\d+)\.", stripped).group(1)
            story.append(Paragraph(f"{num}. {parse_inline(text)}", S["bullet"]))
            i += 1
            continue

        # ── bold-prefixed key point e.g. "**Key:**" ──
        if stripped.startswith("**") and ":" in stripped:
            story.append(Paragraph(parse_inline(stripped), S["body"]))
            i += 1
            continue

        # ── blank line ──
        if not stripped:
            story.append(Spacer(1, 5))
            i += 1
            continue

        # ── plain paragraph ──
        story.append(Paragraph(parse_inline(stripped), S["body"]))
        i += 1

    return story


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Presentation Outline — Tweet-Driven Alpha",
        author="Confluence Research",
    )

    md_text = MD_PATH.read_text(encoding="utf-8")
    story = md_to_flowables(md_text)
    doc.build(story)
    print(f"PDF written to: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
