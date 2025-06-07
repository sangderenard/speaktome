#!/usr/bin/env bash
# Developer-oriented environment setup
# Runs standard setup then dumps headers, stubs, and key documents

set -euo pipefail

# Run the regular setup script with any provided arguments
bash setup_env.sh "$@"

# Dump python file headers in markdown form
python dump_headers.py speaktome --markdown

# Dump stub blocks across the project
python AGENTS/tools/stubfinder.py speaktome

# Display important documentation
cat AGENTS/AGENT_CONSTITUTION.md
cat AGENTS.md

# Show contributor credits
python AGENTS/tools/list_contributors.py

# Show license
cat LICENSE

# Show coding standards
cat AGENTS/CODING_STANDARDS.md
cat AGENTS/CONTRIBUTING.md
# Show project overview
cat AGENTS/PROJECT_OVERVIEW.md 