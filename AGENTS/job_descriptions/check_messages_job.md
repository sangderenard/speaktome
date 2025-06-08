# Check Messages and Reports Job

Maintain the inbox, outbox, and experience report directories.

1. Review `AGENTS/messages/inbox` for new files and archive them after reading.
2. Ensure `AGENTS/messages/outbox` contains only undelivered messages.
3. Run `python AGENTS/validate_guestbook.py` after adding reports.
4. Delete stale logs or temporary files as needed.
