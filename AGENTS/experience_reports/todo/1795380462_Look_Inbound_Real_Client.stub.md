# Live look-to-inbound pairing

After coordinated review and owner-controlled publish/restart, connect an external MCP client to `/mcp/guest`, wait for `guest_inbound_status`, call `guest_inbound_request`, and verify that the admin sees it. Then start a separate authenticated `/mcp` OAuth connection and confirm the admin approves only its exact pending request. Check rate limit behavior with ordinary traffic before changing the limits.
