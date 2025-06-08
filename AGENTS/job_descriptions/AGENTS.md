# Job Descriptions

This folder hosts repeatable tasks for agents. Each Markdown file defines a job
with clear steps and expectations. Agents should read the relevant job
description before performing work.

To obtain a task at random, run:

```bash
python -m AGENTS.tools.dispense_job
```

The script prints the name of a job description file. Open that file, follow the
steps, then document your results in an experience report. Commit your changes
as usual and run the test suite before opening a pull request.

