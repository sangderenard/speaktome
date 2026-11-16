# MCP HTTPS deployment

After both HTTPS repair reviews land, return the live Turing and SpeakToMe checkouts to their default branches through the sanctioned operator workflow. Run `Update-LiveCheckouts.ps1 -ExportWeb`, restart Apache, and verify the public MCP endpoint. The updater skips feature branches and does not land changes.
