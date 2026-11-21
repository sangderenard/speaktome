# MCP cursor recovery activation

Paired source reviews for Turing and SpeakToMe must land before activation. The owner then updates and reloads the service/connector through the normal process. Verify with an account that enters a map, loses its MCP process, invokes explicit leave in a new process, and can enter again. A concurrent move between current-status read and close must return 409 without closing the cursor.

See ticket `9fa269f3d2b848c18d0cb404f564e58b`, Development case `529834940cf9431e92ff6ff23e64d6a6`, and experience report `1795239860_DOC_MCP_Cursor_Recovery.md`.
