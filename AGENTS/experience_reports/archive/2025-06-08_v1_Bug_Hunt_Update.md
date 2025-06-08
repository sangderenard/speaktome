# Bug Hunt Job Update

**Date/Version:** 2025-06-08 v1
**Title:** Expanded bug hunting job description

## Overview
Expanded `bug_hunting_job.md` to clarify expectations when a bug cannot be fixed
immediately. The job now instructs agents to create a ticket, record a stub using
our standard format, and flag severe issues for human review.

## Prompts
```
expand the job description of bug hunter to include creating a ticket if they can't fix or stub is and creating a stub with the bug's nature according to the stub style, we don't want people thinking they have to fix it or fail. Especially bad issues should be marked for human review
```

Custom instructions emphasized exploring existing documentation and keeping trace
in the guest book.

## Steps Taken
1. Reviewed `AGENTS/job_descriptions/bug_hunting_job.md` and repository guidance.
2. Updated the job description with new steps for tickets, stubs, and human review.
3. Created this experience report and ran guestbook validation.
4. Ran the full test suite to ensure no regressions.

## Observed Behaviour
All tests passed. Guestbook validation succeeded with no adjustments.

## Lessons Learned
The repository strongly encourages leaving breadcrumbs in the guestbook for every
change. Clear instructions help new agents understand they should escalate rather
than stall when a bug is too complex.

## Next Steps
Continue to improve job descriptions and explore other documentation for clarity.
