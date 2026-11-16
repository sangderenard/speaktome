# Guest MCP tunnel verification

After the Turing host session-cap change is reviewed and landed, restart the stable service using its existing control request. Repeat a read-only ChatGPT call through the connected “SpeakToMe guest” plugin and confirm `guest_tasks` succeeds. Verify the tunnel runtime is ready after the next machine restart.
