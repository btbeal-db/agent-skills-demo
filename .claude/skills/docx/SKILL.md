---
name: docx
description: "Use this skill whenever the user wants to create, read, edit, or manipulate Word documents (.docx files). Triggers include: any mention of \"Word doc\", \"word document\", \".docx\", or requests to produce professional documents with formatting like tables of contents, headings, page numbers, or letterheads. Also use when extracting or reorganizing content from .docx files, inserting or replacing images in documents, performing find-and-replace in Word files, working with tracked changes or comments, or converting content into a polished Word document. If the user asks for a \"report\", \"memo\", \"letter\", \"template\", or similar deliverable as a Word or .docx file, use this skill. Do NOT use for PDFs, spreadsheets, Google Docs, or general coding tasks unrelated to document generation."
---

# DOCX creation, editing, and analysis

## Overview

A .docx file is a ZIP archive containing XML files.

## Quick Reference

| Task | Approach |
|------|----------|
| Read/analyze content | `mammoth` (Python) or unpack for raw XML |
| Create new document | Use `python-docx` via `execute_python` - see Creating New Documents below |
| Edit existing document | Unpack → edit XML → repack - see Editing Existing Documents below |

### Reading Content

```python
# Text extraction to markdown (via execute_python)
import mammoth
with open("document.docx", "rb") as f:
    result = mammoth.convert_to_markdown(f)
    text = result.value
```

```bash
# Raw XML access
python scripts/office/unpack.py document.docx unpacked/
```

### Converting to Images

```python
# PDF to image conversion (via execute_python)
import fitz  # PyMuPDF
doc = fitz.open("document.pdf")
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    pix.save(f"page-{i+1}.png")
```

---

## Creating New Documents

Generate .docx files with `python-docx` via the `execute_python` tool. The `python-docx` library is already installed.

### Basic Setup
```python
from docx import Document
from docx.shared import Pt, Inches, Cm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
import base64
from io import BytesIO

doc = Document()

# Set default font
style = doc.styles['Normal']
font = style.font
font.name = 'Arial'
font.size = Pt(12)

# Add content...
# Save to buffer for base64 encoding
buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

### Page Size

```python
from docx.shared import Inches

# IMPORTANT: python-docx defaults may vary; always set explicitly
section = doc.sections[0]
section.page_width = Inches(8.5)   # US Letter
section.page_height = Inches(11)
section.top_margin = Inches(1)
section.bottom_margin = Inches(1)
section.left_margin = Inches(1)
section.right_margin = Inches(1)
```

**Common page sizes:**

| Paper | Width | Height |
|-------|-------|--------|
| US Letter | Inches(8.5) | Inches(11) |
| A4 | Cm(21) | Cm(29.7) |

**Landscape orientation:**
```python
from docx.enum.section import WD_ORIENT
section = doc.sections[0]
section.orientation = WD_ORIENT.LANDSCAPE
section.page_width = Inches(11)
section.page_height = Inches(8.5)
```

### Styles (Headings)

```python
doc.add_heading('Title', 0)  # Title style
doc.add_heading('Heading 1', level=1)
doc.add_heading('Heading 2', level=2)

# Custom styled paragraph
para = doc.add_paragraph()
run = para.add_run('Bold text')
run.bold = True
run.font.size = Pt(14)
run.font.name = 'Arial'
```

### Lists

```python
# Bullet list
doc.add_paragraph('First item', style='List Bullet')
doc.add_paragraph('Second item', style='List Bullet')

# Numbered list
doc.add_paragraph('Step one', style='List Number')
doc.add_paragraph('Step two', style='List Number')
```

### Tables

```python
from docx.shared import Inches, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

table = doc.add_table(rows=3, cols=3)
table.style = 'Table Grid'

# Header row
header_cells = table.rows[0].cells
header_cells[0].text = 'Column 1'
header_cells[1].text = 'Column 2'
header_cells[2].text = 'Column 3'

# Data rows
for i, row_data in enumerate([('A', 'B', 'C'), ('D', 'E', 'F')], start=1):
    for j, val in enumerate(row_data):
        table.rows[i].cells[j].text = val

# Set column widths
for row in table.rows:
    row.cells[0].width = Inches(3)
    row.cells[1].width = Inches(3)
    row.cells[2].width = Inches(3)
```

### Images

```python
from docx.shared import Inches

doc.add_picture('image.png', width=Inches(4))
# Or from BytesIO
# doc.add_picture(image_buffer, width=Inches(4))
```

### Page Breaks

```python
doc.add_page_break()

# Or within a paragraph
para = doc.add_paragraph()
run = para.add_run()
run.add_break(WD_BREAK.PAGE)
```

### Headers/Footers

```python
section = doc.sections[0]

# Header
header = section.header
header_para = header.paragraphs[0]
header_para.text = "Document Header"

# Footer with page number
footer = section.footer
footer_para = footer.paragraphs[0]
footer_para.text = "Page "
# Add page number field
run = footer_para.add_run()
fldChar1 = OxmlElement('w:fldChar')
fldChar1.set(qn('w:fldCharType'), 'begin')
run._r.append(fldChar1)
instrText = OxmlElement('w:instrText')
instrText.set(qn('xml:space'), 'preserve')
instrText.text = " PAGE "
run._r.append(instrText)
fldChar2 = OxmlElement('w:fldChar')
fldChar2.set(qn('w:fldCharType'), 'end')
run._r.append(fldChar2)
```

### Next Step After Creation

Once `execute_python` returns successfully (result reported as `<N bytes>` or `<N chars>`), **immediately call `save_to_volume` with just the `filename` parameter**. The result is passed through automatically. Do not add validation or intermediate steps.

### Validation (Editing workflow only)

Validation applies to the **XML editing workflow**, not to `python-docx`-created documents. Do not validate after `execute_python` document creation. If you do need to validate an edited file, run:
```bash
python scripts/office/validate.py doc.docx
```

---

## Editing Existing Documents

**Follow all 3 steps in order.**

### Step 1: Unpack
```bash
python scripts/office/unpack.py document.docx unpacked/
```
Extracts XML, pretty-prints, merges adjacent runs, and converts smart quotes to XML entities (`&#x201C;` etc.) so they survive editing. Use `--merge-runs false` to skip run merging.

### Step 2: Edit XML

Edit files in `unpacked/word/`. See XML Reference below for patterns.

**Use "Claude" as the author** for tracked changes and comments, unless the user explicitly requests use of a different name.

**CRITICAL: Use smart quotes for new content.** When adding text with apostrophes or quotes, use XML entities to produce smart quotes:
```xml
<!-- Use these entities for professional typography -->
<w:t>Here&#x2019;s a quote: &#x201C;Hello&#x201D;</w:t>
```
| Entity | Character |
|--------|-----------|
| `&#x2018;` | ' (left single) |
| `&#x2019;` | ' (right single / apostrophe) |
| `&#x201C;` | " (left double) |
| `&#x201D;` | " (right double) |

**Adding comments:** Use `comment.py` to handle boilerplate across multiple XML files (text must be pre-escaped XML):
```bash
python scripts/comment.py unpacked/ 0 "Comment text with &amp; and &#x2019;"
python scripts/comment.py unpacked/ 1 "Reply text" --parent 0  # reply to comment 0
python scripts/comment.py unpacked/ 0 "Text" --author "Custom Author"  # custom author name
```
Then add markers to document.xml (see Comments in XML Reference).

### Step 3: Pack
```bash
python scripts/office/pack.py unpacked/ output.docx --original document.docx
```
Validates with auto-repair, condenses XML, and creates DOCX. Use `--validate false` to skip.

**Auto-repair will fix:**
- `durableId` >= 0x7FFFFFFF (regenerates valid ID)
- Missing `xml:space="preserve"` on `<w:t>` with whitespace

**Auto-repair won't fix:**
- Malformed XML, invalid element nesting, missing relationships, schema violations

### Common Pitfalls

- **Replace entire `<w:r>` elements**: When adding tracked changes, replace the whole `<w:r>...</w:r>` block with `<w:del>...<w:ins>...` as siblings. Don't inject tracked change tags inside a run.
- **Preserve `<w:rPr>` formatting**: Copy the original run's `<w:rPr>` block into your tracked change runs to maintain bold, font size, etc.

---

## XML Reference

### Schema Compliance

- **Element order in `<w:pPr>`**: `<w:pStyle>`, `<w:numPr>`, `<w:spacing>`, `<w:ind>`, `<w:jc>`, `<w:rPr>` last
- **Whitespace**: Add `xml:space="preserve"` to `<w:t>` with leading/trailing spaces
- **RSIDs**: Must be 8-digit hex (e.g., `00AB1234`)

### Tracked Changes

**Insertion:**
```xml
<w:ins w:id="1" w:author="Claude" w:date="2025-01-01T00:00:00Z">
  <w:r><w:t>inserted text</w:t></w:r>
</w:ins>
```

**Deletion:**
```xml
<w:del w:id="2" w:author="Claude" w:date="2025-01-01T00:00:00Z">
  <w:r><w:delText>deleted text</w:delText></w:r>
</w:del>
```

**Inside `<w:del>`**: Use `<w:delText>` instead of `<w:t>`, and `<w:delInstrText>` instead of `<w:instrText>`.

**Minimal edits** - only mark what changes:
```xml
<!-- Change "30 days" to "60 days" -->
<w:r><w:t>The term is </w:t></w:r>
<w:del w:id="1" w:author="Claude" w:date="...">
  <w:r><w:delText>30</w:delText></w:r>
</w:del>
<w:ins w:id="2" w:author="Claude" w:date="...">
  <w:r><w:t>60</w:t></w:r>
</w:ins>
<w:r><w:t> days.</w:t></w:r>
```

**Deleting entire paragraphs/list items** - when removing ALL content from a paragraph, also mark the paragraph mark as deleted so it merges with the next paragraph. Add `<w:del/>` inside `<w:pPr><w:rPr>`:
```xml
<w:p>
  <w:pPr>
    <w:numPr>...</w:numPr>  <!-- list numbering if present -->
    <w:rPr>
      <w:del w:id="1" w:author="Claude" w:date="2025-01-01T00:00:00Z"/>
    </w:rPr>
  </w:pPr>
  <w:del w:id="2" w:author="Claude" w:date="2025-01-01T00:00:00Z">
    <w:r><w:delText>Entire paragraph content being deleted...</w:delText></w:r>
  </w:del>
</w:p>
```
Without the `<w:del/>` in `<w:pPr><w:rPr>`, accepting changes leaves an empty paragraph/list item.

**Rejecting another author's insertion** - nest deletion inside their insertion:
```xml
<w:ins w:author="Jane" w:id="5">
  <w:del w:author="Claude" w:id="10">
    <w:r><w:delText>their inserted text</w:delText></w:r>
  </w:del>
</w:ins>
```

**Restoring another author's deletion** - add insertion after (don't modify their deletion):
```xml
<w:del w:author="Jane" w:id="5">
  <w:r><w:delText>deleted text</w:delText></w:r>
</w:del>
<w:ins w:author="Claude" w:id="10">
  <w:r><w:t>deleted text</w:t></w:r>
</w:ins>
```

### Comments

After running `comment.py` (see Step 2), add markers to document.xml. For replies, use `--parent` flag and nest markers inside the parent's.

**CRITICAL: `<w:commentRangeStart>` and `<w:commentRangeEnd>` are siblings of `<w:r>`, never inside `<w:r>`.**

```xml
<!-- Comment markers are direct children of w:p, never inside w:r -->
<w:commentRangeStart w:id="0"/>
<w:del w:id="1" w:author="Claude" w:date="2025-01-01T00:00:00Z">
  <w:r><w:delText>deleted</w:delText></w:r>
</w:del>
<w:r><w:t> more text</w:t></w:r>
<w:commentRangeEnd w:id="0"/>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="0"/></w:r>

<!-- Comment 0 with reply 1 nested inside -->
<w:commentRangeStart w:id="0"/>
  <w:commentRangeStart w:id="1"/>
  <w:r><w:t>text</w:t></w:r>
  <w:commentRangeEnd w:id="1"/>
<w:commentRangeEnd w:id="0"/>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="0"/></w:r>
<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:commentReference w:id="1"/></w:r>
```

### Images

1. Add image file to `word/media/`
2. Add relationship to `word/_rels/document.xml.rels`:
```xml
<Relationship Id="rId5" Type=".../image" Target="media/image1.png"/>
```
3. Add content type to `[Content_Types].xml`:
```xml
<Default Extension="png" ContentType="image/png"/>
```
4. Reference in document.xml:
```xml
<w:drawing>
  <wp:inline>
    <wp:extent cx="914400" cy="914400"/>  <!-- EMUs: 914400 = 1 inch -->
    <a:graphic>
      <a:graphicData uri=".../picture">
        <pic:pic>
          <pic:blipFill><a:blip r:embed="rId5"/></pic:blipFill>
        </pic:pic>
      </a:graphicData>
    </a:graphic>
  </wp:inline>
</w:drawing>
```

---

## Dependencies

- **mammoth**: `pip install mammoth` (text extraction to markdown, replaces pandoc)
- **python-docx**: `pip install python-docx` (new document creation, replaces docx-js)
- **PyMuPDF**: `pip install PyMuPDF` (PDF to image conversion, replaces poppler-utils)

> **Note**: LibreOffice-based operations (accepting tracked changes via `accept_changes.py`, `.doc` to `.docx` conversion) are not available in this environment. Use the unpack/edit/repack workflow for tracked change editing instead.

---

## Runtime Adaptation (Databricks Agent)

This skill runs inside a Databricks agent with these tools: `execute_python`, `execute_bash`, `save_to_volume`, `read_from_volume`, `list_volume_files`.

### Important: Script Paths

When you call `load_skill`, the response includes `Scripts path: <absolute_path>`. **Always use this absolute path** when running helper scripts via `execute_bash`:
```bash
python <scripts_path>/office/unpack.py document.docx unpacked/
python <scripts_path>/office/pack.py unpacked/ output.docx --original document.docx
python <scripts_path>/comment.py unpacked/ 0 "Comment text"
python <scripts_path>/office/validate.py output.docx
```

### UC Volume Bridge Workflow

All files must be persisted to Unity Catalog Volumes. Use this pattern:

**Reading a file from UC Volume for processing:**
1. `read_from_volume` — loads file content (available as `source_doc_bytes`)
2. `execute_python` — write bytes to a local file in the working directory:
   ```python
   with open("/tmp/agent_bash_xxx/document.docx", "wb") as f:
       f.write(source_doc_bytes)
   result = "File written to working directory"
   ```
3. `execute_bash` — process with scripts (unpack, edit, pack, etc.)

**Saving a processed file to UC Volume:**
1. `execute_python` — read the output file and base64-encode:
   ```python
   import base64
   with open("/tmp/agent_bash_xxx/output.docx", "rb") as f:
       result = base64.b64encode(f.read()).decode('utf-8')
   ```
2. `save_to_volume` — persist with filename

**Creating a new document (no input file):**
1. `execute_python` — use `python-docx` to create the document, base64-encode the result
2. `save_to_volume` — persist with filename (pass only `filename`; result bytes are passed through automatically)

**After step 1 succeeds, go directly to step 2. Do not validate, do not add intermediate steps.**

**Reading a document for text extraction:**
1. `read_from_volume` — loads file content
2. `execute_python` — use `mammoth` to extract text:
   ```python
   import mammoth
   from io import BytesIO
   with BytesIO(source_doc_bytes) as f:
       result = mammoth.convert_to_markdown(f).value
   ```
