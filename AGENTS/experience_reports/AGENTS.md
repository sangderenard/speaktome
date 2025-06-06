# Experience Reports Guide

This directory stores guestbook entries created by visiting agents. Each file should follow the naming convention described in `AGENTS/GUESTBOOK.md`:

```
YYYY-MM-DD_v<version>_Descriptive_Title.md
```

Include a **Prompt History** section quoting any instructions or conversations that influenced the session verbatim. This helps future agents understand why a given exploration was performed.

After adding or updating a report, run `python AGENTS/validate_guestbook.py` to confirm filenames conform and archives are updated automatically.

