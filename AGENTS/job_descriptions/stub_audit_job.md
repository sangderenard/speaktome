# Stub Audit Job

Verify that each file listed in `AGENTS/stub_audit_list.txt` adheres to
`AGENTS/CODING_STANDARDS.md` for stub formatting. For each file:

1. Open the file and search for `########## STUB` markers.
2. Confirm the block comment includes PURPOSE, EXPECTED BEHAVIOR, INPUTS,
   OUTPUTS, KEY ASSUMPTIONS, TODO, and NOTES sections.
3. If the file has no stubs, mark it `OK`.
4. Record your check in `AGENTS/stub_audit_signoff.txt` with `OK` or details.
5. Commit updates and run the test suite using `./.venv/bin/pytest -q` (Windows: `.venv\Scripts\pytest.exe -q`).
