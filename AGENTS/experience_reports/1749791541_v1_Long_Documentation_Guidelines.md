# Long Documentation Guidelines

**Date/Version:** 1749791541 v1
**Title:** Special handling for lengthy reports

## Overview
Some development tasks produce documentation far longer than the typical guestbook template. This record establishes guidance for writing "long form" experience reports when a short summary cannot capture the required detail.

## Prompts
- "add a document to experience reports on long documentation, that some tasks may require documentation significantly longer than the template implies is acceptable"

## Long Report Types
1. **Problem Reports** – exhaustive breakdowns of failures or bugs, including every command and log snippet that led to the issue.
2. **Systems Analysis Reports** – in-depth explanations of how a feature behaves in the actual code, line by line, rather than relying on high level design notes.

## Special Instructions
- Start with a concise abstract so readers know the context of the long report.
- Use section headings liberally to break up content.
- Include full command transcripts or code excerpts when they illuminate the analysis.
- When referencing prompts or instructions, quote them exactly as in any other report.
- Close with actionable conclusions or next steps, even if the document spans many pages.

## Template
Long reports may diverge from the standard template but should include at least:
- Metadata block with date, version, and title
- Prompt history
- Step‑by‑step chronology of actions taken
- Detailed observations and code analysis
- Takeaways and follow‑up actions

The filename should still follow the `EPOCH_v<version>_Descriptive_Title.md` convention so the validation script can index it.

## Next Steps
Future agents encountering oversized documentation needs can mirror this template and adjust section headings as required.
