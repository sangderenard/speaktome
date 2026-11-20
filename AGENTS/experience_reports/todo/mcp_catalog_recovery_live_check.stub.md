# MCP catalog recovery live check

After review lands and the MCP process reloads, let the MCP server encounter a temporary `/api/index` outage. When the service recovers, request `tools/list` or `api_index` and verify it adopts the service index and removes any tools not allowed for the account. A client that caches tools may need to reconnect.

Source: `../1795201372_LOG_MCP_Catalog_Recovery_After_503.md`.
