# Experience Reports Guide

Welcome to the guestbook kiosk of our agent theme park. Here every report is a journal entry about your tour. Each file should follow one of the category-based naming conventions described in `AGENTS/GUESTBOOK.md`:

```
EPOCH_DOC_Descriptive_Title.md
EPOCH_TTICKET_Descriptive_Title.md
EPOCH_AUDIT_Descriptive_Title.md
EPOCH_LOG_Descriptive_Title.md
```

Templates for each category are available:

- `template_doc_report.md`
- `template_tticket_report.md`
- `template_audit_report.md`
- `template_log_report.md`

Include a **Prompt History** section quoting any instructions or conversations that influenced the session verbatim. Think of it as leaving breadcrumbs on the trail so others can retrace your route.

After adding or updating a report, run `python AGENTS/validate_guestbook.py` to confirm filenames conform and archives are updated automatically. Capture any "Next Steps" from your report as a `.stub.md` file in `todo/` so future agents can quickly locate outstanding tasks. This keeps the park map tidy and easy to read.


For unusually detailed analyses, create a "long documentation" report. See `1749791541_v1_Long_Documentation_Guidelines.md` for the recommended format and extended instructions.

