---
name: pdf
description: "Use this skill whenever the user wants to read, extract text from, fill in blanks, annotate, redact, merge, split, rotate, watermark, or otherwise work with PDF files. Triggers include: any mention of \".pdf\", requests to fill in form fields or blanks in a PDF, extract text or tables from a PDF, add comments or highlights to a PDF, combine or split PDF pages, redact sensitive content, or add a watermark. Do NOT use for Word documents (.docx), spreadsheets, or operations unrelated to PDF content."
---

# PDF reading, editing, and manipulation

## Overview

PDFs are not reflowable documents like DOCX — content is rendered at fixed positions. Editing is done by locating content (by text search or coordinates), whiting it out, and overlaying replacements. All operations use `fitz` (PyMuPDF), which is already installed.

## Quick Reference

| Task | Approach |
|------|----------|
| Extract all text | `page.get_text()` |
| Find text location | `page.search_for("text")` → list of `Rect` |
| Fill blank / replace text | `add_redact_annot(rect, new_text)` + `apply_redactions()` |
| Detect blank type | See **Detecting Blank Types** below |
| Highlight / underline / strikeout | `page.add_highlight_annot(rect)` etc. |
| Add sticky-note comment | `page.add_text_annot(point, "note")` |
| Redact (remove permanently) | `add_redact_annot(rect)` + `apply_redactions()` |
| Watermark | `page.insert_text(point, text, ...)` |
| Merge PDFs | `doc.insert_pdf(src)` |
| Split pages | `doc.select([page_numbers])` |
| Rotate pages | `page.set_rotation(degrees)` |
| Read metadata | `doc.metadata` |

---

## Reading Content

```python
# Text extraction (via execute_python)
import fitz
from io import BytesIO

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")

# All text
text = "\n\n".join(page.get_text() for page in doc)
result = text
```

```python
# Structured blocks with position info
import fitz
from io import BytesIO

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")
page = doc[0]

# Each block: (x0, y0, x1, y1, text, block_no, block_type)
blocks = page.get_text("blocks")
for b in blocks:
    print(f"[{b[0]:.0f},{b[1]:.0f}] {b[4][:80]}")
result = str(blocks)
```

```python
# Page count and metadata
import fitz
from io import BytesIO

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")
result = f"Pages: {doc.page_count}\nMetadata: {doc.metadata}"
```

---

## Filling Blanks

PDFs use different techniques for blanks. Identify the type first, then use the matching fill approach.

### Detecting Blank Types

```python
# Run this on an unfamiliar PDF to identify blank representation
import fitz
from io import BytesIO

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")
page = doc[0]

# Check for AcroForm fields
widgets = list(page.widgets())
print(f"Form fields (AcroForm): {len(widgets)}")
for w in widgets:
    print(f"  field={w.field_name!r}, value={w.field_value!r}, rect={w.rect}")

# Check for underscore/placeholder text
for pattern in ["___", "____", "______", "__________"]:
    hits = page.search_for(pattern)
    if hits:
        print(f"Text blank ({pattern!r}): {len(hits)} found")

# Check for drawn lines (visual blanks)
drawings = page.get_drawings()
lines = [d for d in drawings if d["type"] == "l"]
print(f"Drawn lines: {len(lines)}")
result = "Blank detection complete — check output above"
```

### Option 1: AcroForm Fields (true PDF form fields)

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")

# Fill by field name
for page in doc:
    for widget in page.widgets():
        name = widget.field_name
        if name == "UserName":
            widget.field_value = "Brennan Beal"
            widget.update()
        elif name == "Date":
            widget.field_value = "2025-01-01"
            widget.update()

buf = BytesIO()
doc.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

### Option 2: Underscore / Placeholder Text Blanks

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")

# Map of placeholder → replacement
fills = {
    "___": "Brennan Beal",      # adjust pattern to match actual blank
}

for page in doc:
    for placeholder, replacement in fills.items():
        for rect in page.search_for(placeholder):
            # White out the blank and insert replacement text
            page.add_redact_annot(rect, replacement, fontsize=11, align=fitz.TEXT_ALIGN_LEFT)
    page.apply_redactions()

buf = BytesIO()
doc.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

**Tips:**
- Match the exact placeholder string (check with `page.search_for(pattern)` first)
- Adjust `fontsize` to match the surrounding text — extract nearby text blocks to infer size
- Use `align=fitz.TEXT_ALIGN_CENTER` for centered fields
- If a label precedes the blank (e.g., `"Name: ___"`), search for the full label to anchor position

### Option 3: Drawn-Line Blanks (insert text above a line)

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")
page = doc[0]

# Find horizontal lines that are likely blank fields
drawings = page.get_drawings()
fill_lines = [
    d for d in drawings
    if d["type"] == "l"
    and abs(d["rect"].y1 - d["rect"].y0) < 3   # nearly horizontal
    and d["rect"].width > 50                     # wide enough to be a field
]

fills = ["Brennan Beal", "2025-01-01"]  # ordered to match line positions top-to-bottom
fill_lines.sort(key=lambda d: (d["rect"].y0, d["rect"].x0))

for line, text in zip(fill_lines, fills):
    r = line["rect"]
    # Insert text just above the line
    page.insert_text((r.x0, r.y0 - 2), text, fontsize=11)

buf = BytesIO()
doc.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

---

## Annotations

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")
page = doc[0]

# Highlight text
for rect in page.search_for("important phrase"):
    annot = page.add_highlight_annot(rect)
    annot.update()

# Underline text
for rect in page.search_for("underlined text"):
    annot = page.add_underline_annot(rect)
    annot.update()

# Strikeout
for rect in page.search_for("removed text"):
    annot = page.add_strikeout_annot(rect)
    annot.update()

# Sticky-note comment at a position
annot = page.add_text_annot((100, 100), "Review this section", icon="Note")
annot.set_info(title="Claude", content="This needs legal review")
annot.update()

buf = BytesIO()
doc.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

---

## Redaction (Permanent Removal)

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")

for page in doc:
    # Search and permanently black out
    for rect in page.search_for("CONFIDENTIAL"):
        page.add_redact_annot(rect)   # no replacement text = black box
    page.apply_redactions()

buf = BytesIO()
doc.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

---

## Watermarks

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")

for page in doc:
    rect = page.rect
    # Insert diagonal watermark text
    page.insert_text(
        (rect.width * 0.15, rect.height * 0.55),
        "DRAFT",
        fontsize=72,
        color=(0.75, 0.75, 0.75),  # light grey
        rotate=45,
    )

buf = BytesIO()
doc.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

---

## Page Manipulation

### Merge multiple PDFs

```python
import fitz
from io import BytesIO
import base64

# source_doc_bytes must be a list of bytes objects, one per PDF
# If reading multiple files, use read_from_volume for each and collect into a list

combined = fitz.open()
for pdf_bytes in source_doc_bytes_list:
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    combined.insert_pdf(src)

buf = BytesIO()
combined.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

### Split: extract specific pages

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")

# Keep only pages 0, 2, 4 (0-indexed)
doc.select([0, 2, 4])

buf = BytesIO()
doc.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

### Rotate pages

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")

# Rotate all pages 90° clockwise
for page in doc:
    page.set_rotation(90)

# Or rotate a specific page
doc[0].set_rotation(180)

buf = BytesIO()
doc.save(buf)
buf.seek(0)
result = base64.b64encode(buf.read()).decode()
```

### Render pages to images

```python
import fitz
from io import BytesIO
import base64

doc = fitz.open(stream=source_doc_bytes, filetype="pdf")

images = {}
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    images[f"page-{i+1}.png"] = base64.b64encode(pix.tobytes("png")).decode()

result = f"Rendered {len(images)} pages"
# Save individual pages via save_to_volume with each image's base64 data
```

---

## Dependencies

- **PyMuPDF** (`fitz`): already installed — covers all operations above
- No additional packages required

---

## Runtime Adaptation (Databricks Agent)

This skill runs inside a Databricks agent with tools: `execute_python`, `execute_bash`, `save_to_volume`, `read_from_volume`, `list_volume_files`.

### UC Volume Bridge Workflow

**Reading a PDF from UC Volume for processing:**
1. `read_from_volume` — loads file bytes (available as `source_doc_bytes` in `execute_python`)
2. `execute_python` — open directly from bytes (no temp file needed for fitz):
   ```python
   import fitz
   doc = fitz.open(stream=source_doc_bytes, filetype="pdf")
   ```

**Saving a processed PDF to UC Volume:**
1. `execute_python` — process the document, then base64-encode the result:
   ```python
   from io import BytesIO
   import base64
   buf = BytesIO()
   doc.save(buf)
   buf.seek(0)
   result = base64.b64encode(buf.read()).decode()
   ```
2. `save_to_volume` — persist with filename (pass only `filename`; result bytes pass through automatically)

**After step 1 succeeds, go directly to step 2. Do not add intermediate validation steps.**

### Blank-Filling Recommended Workflow

1. `read_from_volume` — load the PDF
2. `execute_python` — run blank detection to identify blank type (see **Detecting Blank Types**)
3. Ask the user to confirm field mappings if not already specified
4. `execute_python` — apply fills using the matching option (AcroForm / text / drawn-line)
5. `save_to_volume` — persist the filled PDF
