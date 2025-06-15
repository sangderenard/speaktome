# User Experience Report Guidelines

This folder contains all user experience reports and supporting files. Use these guidelines when adding new reports. Treat the experience reports as a centralized log of automated instructions and AI-assisted experiments.

We encourage **four complementary reporting modes**:

1. **DOC** — documentation of repository activity.
2. **TTICKET** — trouble tickets describing errors or unexpected behaviour.
3. **AUDIT** — in‑depth auditing of functions and design decisions.
4. **LOG** — raw logs captured from repeatable commands.

## Naming Convention

Store reports in `experience_reports/`. Choose a category and name the file as:

```
EPOCH_DOC_Descriptive_Title.md
EPOCH_TTICKET_Descriptive_Title.md
EPOCH_AUDIT_Descriptive_Title.md
EPOCH_LOG_Descriptive_Title.md
```

* `EPOCH` is the timestamp or date of the entry.
* `DOC`, `TTICKET`, `AUDIT`, or `LOG` indicate the level of detail:
  * **DOC** — brief activity notes.
  * **TTICKET** — trouble tickets describing errors.
  * **AUDIT** — in‑depth systematic explorations.
  * **LOG** — raw log dumps for debugging or reference.
* `Descriptive_Title` summarizes the scenario using `_` instead of spaces.

Example: `1720123456_DOC_New_User_Experience_Simulation.md`.

## Role‑Playing Exercise

When writing a report, take the perspective of a new user exploring the repository. Document your attempts, commands, and observations as if you were encountering the project for the first time. Compare with previous reports, but avoid repeating identical steps that did not produce new insights. Instead, adjust your approach and note what changed.

## Learning from Past Reports

Review earlier experiences before starting a new test. Identify unresolved questions or unexplored features and focus on those areas. Summarize how your new approach differs so the team can track progress without duplicating effort.

## Continuous Collaboration

User experience reports are part of a dynamic feedback loop between developers and testers. Keep the conversation going by clearly stating next steps and open issues. Whenever you outline follow-up tasks, also create a `.stub.md` file in `todo/` summarizing the action. This makes it easier for future contributors to find and implement outstanding work. The goal is incremental refinement of the project.

## Prompt History

Include a section in each report that captures verbatim any prompts or scripted instructions that guided the session. This record helps future LLM agents quickly understand the context and thought process behind your experiments.

## Templates

Use the matching template for your report type:

- `experience_reports/template_doc_report.md`
- `experience_reports/template_tticket_report.md`
- `experience_reports/template_audit_report.md`
- `experience_reports/template_log_report.md`

