# User Experience Report Guidelines

This folder contains all user experience reports and supporting files. Use these guidelines when adding new reports. Treat the experience reports as a centralized log of automated instructions and AI-assisted experiments.

## Naming Convention

Store reports in `experience_reports/`. Name each file as:

```
EPOCH_v<version>_Descriptive_Title.md
```

* `EPOCH` is the date of the experiment.
* `v<version>` increments when multiple iterations occur on the same day.
* `Descriptive_Title` summarizes the scenario using `_` instead of spaces.

Example: `1720123456_v1_New_User_Experience_Simulation.md`.

## Role‑Playing Exercise

When writing a report, take the perspective of a new user exploring the repository. Document your attempts, commands, and observations as if you were encountering the project for the first time. Compare with previous reports, but avoid repeating identical steps that did not produce new insights. Instead, adjust your approach and note what changed.

## Learning from Past Reports

Review earlier experiences before starting a new test. Identify unresolved questions or unexplored features and focus on those areas. Summarize how your new approach differs so the team can track progress without duplicating effort.

## Continuous Collaboration

User experience reports are part of a dynamic feedback loop between developers and testers. Keep the conversation going by clearly stating next steps and open issues. The goal is incremental refinement of the project.

## Prompt History

Include a section in each report that captures verbatim any prompts or scripted instructions that guided the session. This record helps future LLM agents quickly understand the context and thought process behind your experiments.

## Template

Use `experience_reports/template_experience_report.md` as a starting point for new documents.

