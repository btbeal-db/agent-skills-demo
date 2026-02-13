---
name: docx
description: "Use this skill to create, read, or edit Word documents (.docx files) using Python. Triggers include: mentions of 'Word doc', 'word document', '.docx', or requests to produce professional documents with formatting. Uses python-docx library. Saves output to Unity Catalog Volume."
---

# DOCX Creation, Editing, and Analysis (Python-Only)

This skill uses `python-docx` for all Word document operations. All generated documents are saved to a Unity Catalog Volume.

## Quick Reference

| Task | Approach |
|------|----------|
| Create new document | Use `python-docx` with `execute_python` tool |
| Read document | Load with `Document(path)` |
| Edit document | Load, modify, save |
| Save to UC Volume | Use `save_to_volume` tool with base64 content |

## Import Requirements

When using `execute_python`, no third-party modules are pre-imported. Include required imports in your code block, for example:

```python
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from io import BytesIO
from pathlib import Path
```

---

## Creating Documents

### Basic Document

```python
from docx import Document
import base64
from io import BytesIO

# Create a simple document
doc = Document()

# Add a title
doc.add_heading('My Report', 0)

# Add a paragraph
doc.add_paragraph('This is the introduction to my report.')

# Add a heading
doc.add_heading('Section 1', level=1)
doc.add_paragraph('Content for section 1.')

# Save to BytesIO for base64 encoding
buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

Then use `save_to_volume` with the base64 result.

### Document with Formatting

```python
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import base64
from io import BytesIO

doc = Document()

# Title with center alignment
title = doc.add_heading('Quarterly Report', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Styled paragraph
para = doc.add_paragraph()
run = para.add_run('Bold and ')
run.bold = True
run = para.add_run('italic text.')
run.italic = True

# Paragraph with specific font size
para = doc.add_paragraph()
run = para.add_run('This is 14pt text.')
run.font.size = Pt(14)

# Save
buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

### Lists

```python
from docx import Document
import base64
from io import BytesIO

doc = Document()
doc.add_heading('My List', level=1)

# Bullet list
doc.add_paragraph('First item', style='List Bullet')
doc.add_paragraph('Second item', style='List Bullet')
doc.add_paragraph('Third item', style='List Bullet')

# Numbered list
doc.add_heading('Numbered List', level=2)
doc.add_paragraph('Step one', style='List Number')
doc.add_paragraph('Step two', style='List Number')
doc.add_paragraph('Step three', style='List Number')

buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

### Tables

```python
from docx import Document
from docx.shared import Inches
import base64
from io import BytesIO

doc = Document()
doc.add_heading('Data Table', level=1)

# Create a table with 3 rows and 3 columns
table = doc.add_table(rows=3, cols=3)
table.style = 'Table Grid'

# Header row
header_cells = table.rows[0].cells
header_cells[0].text = 'Name'
header_cells[1].text = 'Department'
header_cells[2].text = 'Role'

# Data rows
data = [
    ('Alice', 'Engineering', 'Developer'),
    ('Bob', 'Marketing', 'Manager'),
]

for i, (name, dept, role) in enumerate(data, start=1):
    row_cells = table.rows[i].cells
    row_cells[0].text = name
    row_cells[1].text = dept
    row_cells[2].text = role

buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

### Images

```python
from docx import Document
from docx.shared import Inches
import base64
from io import BytesIO

doc = Document()
doc.add_heading('Document with Image', level=1)

# Add image from file path
doc.add_picture('/path/to/image.png', width=Inches(4))

# Or add image from BytesIO
# image_buffer = BytesIO(image_bytes)
# doc.add_picture(image_buffer, width=Inches(4))

doc.add_paragraph('Caption: My image above.')

buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

### Page Breaks

```python
from docx import Document
from docx.enum.text import WD_BREAK
import base64
from io import BytesIO

doc = Document()

doc.add_heading('Page 1', level=1)
doc.add_paragraph('Content on page 1.')

# Add page break
doc.add_page_break()

doc.add_heading('Page 2', level=1)
doc.add_paragraph('Content on page 2.')

buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

---

## Reading Documents

### Extract All Text

```python
from docx import Document

# Read from UC Volume path or local path
doc = Document('/Volumes/btbeal/docx_agent_skills_demo/created_docs/input.docx')

# Extract all text
full_text = []
for para in doc.paragraphs:
    full_text.append(para.text)

result = '\n'.join(full_text)
```

### Extract Tables

```python
from docx import Document

doc = Document('/path/to/document.docx')

tables_data = []
for table in doc.tables:
    table_data = []
    for row in table.rows:
        row_data = [cell.text for cell in row.cells]
        table_data.append(row_data)
    tables_data.append(table_data)

result = tables_data
```

### Get Document Structure

```python
from docx import Document

doc = Document('/path/to/document.docx')

structure = []
for para in doc.paragraphs:
    style = para.style.name if para.style else 'Normal'
    structure.append({
        'style': style,
        'text': para.text[:100] + '...' if len(para.text) > 100 else para.text
    })

result = structure
```

---

## Editing Documents

### Modify Existing Document

```python
from docx import Document
import base64
from io import BytesIO

# Load existing document
doc = Document('/path/to/existing.docx')

# Add new content
doc.add_heading('New Section', level=1)
doc.add_paragraph('This section was added by the agent.')

# Save modified document
buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

### Find and Replace Text

```python
from docx import Document
import base64
from io import BytesIO

doc = Document('/path/to/document.docx')

# Simple find and replace in paragraphs
for para in doc.paragraphs:
    if 'OLD_TEXT' in para.text:
        for run in para.runs:
            run.text = run.text.replace('OLD_TEXT', 'NEW_TEXT')

buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

---

## Complete Workflow Example

Here's how to create a document and save it to UC Volume:

**Step 1: Execute Python to create document**

```python
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import base64
from io import BytesIO

# Create document
doc = Document()

# Add title
title = doc.add_heading('Project Status Report', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Add date
doc.add_paragraph('Date: February 11, 2026')

# Add summary section
doc.add_heading('Executive Summary', level=1)
doc.add_paragraph(
    'This report summarizes the current status of our project. '
    'All milestones are on track for Q1 completion.'
)

# Add table
doc.add_heading('Milestone Status', level=1)
table = doc.add_table(rows=4, cols=3)
table.style = 'Table Grid'

# Headers
headers = table.rows[0].cells
headers[0].text = 'Milestone'
headers[1].text = 'Status'
headers[2].text = 'Due Date'

# Data
milestones = [
    ('Phase 1', 'Complete', 'Jan 15'),
    ('Phase 2', 'In Progress', 'Feb 28'),
    ('Phase 3', 'Planned', 'Mar 31'),
]

for i, (name, status, due) in enumerate(milestones, start=1):
    row = table.rows[i].cells
    row[0].text = name
    row[1].text = status
    row[2].text = due

# Save to buffer
buffer = BytesIO()
doc.save(buffer)
buffer.seek(0)
result = base64.b64encode(buffer.read()).decode('utf-8')
```

**Step 2: Save to UC Volume**

Use the `save_to_volume` tool with:
- `filename`: "status_report.docx"
- `content_base64`: (the result from step 1)
- `content_type`: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

---

## Tips

1. **Always use BytesIO and base64** - Convert documents to base64 for saving via `save_to_volume`
2. **Use built-in styles** - 'Heading 1', 'Heading 2', 'List Bullet', 'List Number', 'Table Grid'
3. **Units** - Use `Pt()` for points, `Inches()` for inches, `Cm()` for centimeters
4. **Set result variable** - The `execute_python` tool looks for a `result` variable for output

## Common Styles

| Style Name | Use For |
|------------|---------|
| `'Title'` | Document title |
| `'Heading 1'` | Major sections |
| `'Heading 2'` | Subsections |
| `'List Bullet'` | Bullet points |
| `'List Number'` | Numbered lists |
| `'Table Grid'` | Tables with borders |
| `'Quote'` | Block quotes |
