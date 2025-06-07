#!/usr/bin/env bash
set -euo pipefail

echo "🛠 Setting up the environment..."
bash setup_env.sh --extras --prefetch

echo
echo "📜 Showing the agent introduction:"
echo "--------------------------------------"
head -n 20 AGENTS.md
echo "--------------------------------------"
echo

echo "💡 You can explore full agent docs here:"
echo "  less AGENTS.md"
echo
echo "🎬 Running the automated demo (non-interactive):"
bash auto_demo.sh || echo "⚠️ Automated demo encountered an issue, but setup continues."

echo
echo "✅ Ready. You can now:"
echo "  source .venv/bin/activate"
echo "  bash run.sh -s 'Hello' -m 10 -c -a 5 --final_viz"
