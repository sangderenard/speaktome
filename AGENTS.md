********************************************
*       SPEAKTOME MULTI-PROJECT GUIDE      *
********************************************

Welcome, wandering LLM! Think of this repository as an **agent theme park**. Every directory is a trail with its own informational placards. Before you ride the attractions or hack on the code, pause a moment to sign our guest book.

The goal is comfort and orientation. Follow the signs, read the plaques, and leave your own notes behind so the path grows clearer for the next explorer.

Guest book entries live under `AGENTS/experience_reports/`. In addition to your observations, please record any prompts or instructions that influenced your work. Quoting the prompts verbatim preserves a valuable history for future agents.

The `AGENTS/experience_reports` directory hosts its own `AGENTS.md` and a collection of experience reports. Treat this folder as a ledger of those who explored before you. Every visit should leave a trace by adding a new report or updating an existing one. Use the provided template or mirror the established naming pattern.

A helper script `AGENTS/validate_guestbook.py` will scan the guest book and enforce the filename convention. Run it after creating a report to confirm everything is in order. The `AGENTS/experience_reports/AGENTS.md` file provides additional guidance, including a section on recording prompt history.

Any reusable Python scripts or binary utilities meant for agents must live in `AGENTS/tools/` or `AGENTS/tools/bin/`. Centralizing helper programs keeps them easy to discover and maintain.

For deeper historical context, read through all prior reports. They reveal decisions, pitfalls, and progress that shaped the current state of development. If you are tempted to install packages manually with `pip`, stop and read `AGENTS_DO_NOT_PIP_MANUALLY.md` first. The provided setup scripts manage dependencies for you and explain how optional groups work. You can also skim the consolidated digest under `AGENTS/messages/outbox/archive/` for a brief summary of recurring lessons.

**CI and Headless Agents:** invoke `setup_env_dev.sh` without any extras or prefetch flags. Past documentation suggested running `AGENTS/tools/dev_group_menu.py` directly, but this approach proved unreliable. See `AGENTS/OBSOLETE_SETUP_GUIDE.md` for details.

If you crave an immediate, exhaustive overview, run this one-liner. It will spew every markdown, script and source file to your terminal. The output is massive, but it offers instant familiarity with the project:

```bash
bash -c 'echo "Dumping all docs and code (huge output)..." && \
  find . \( -path ./.git -o -path ./.venv \) -prune -o \
  -type f \( -name "*.md" -o -name "*.py" -o -name "*.sh" -o -name "*.ps1" \) \
  -exec echo "===== {} =====" \; -exec cat {} \;'
```

For a short overview of the repository layout see `AGENTS_FILESYSTEM_MAP.md`. Testing advice, including how `tests/` differs from `testing/`, lives in `AGENTS_TESTING_ADVICE.md`.

## Job Selection

Agents unsure what to work on can request a task via the job dispenser:

```bash
python -m AGENTS.tools.dispense_job
```

For an interactive menu that also runs any obvious setup commands, use:

```bash
python -m AGENTS.tools.select_and_run_job
```

Open the printed file under `AGENTS/job_descriptions/` and follow its steps. Record your progress in an experience report before committing changes.

## Available Codebases

- **speaktome** — main beam search controllers and utilities.
- **laplace** — Laplace builder and DEC utilities.
- **tensor_printing** — experimental Grand Printing Press package.
- **time_sync** — system clock offset helpers.
- **AGENTS/tools** — shared helper scripts for repository management.

See `AGENTS/CODEBASE_REGISTRY.md` for details and future additions.
