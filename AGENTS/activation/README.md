# Activation: SpeakToMe messages start Claude

This folder is the whole prompt for the scheduled message check. The owner set it up on 2026-11-13.

The Claude desktop app holds a scheduled task, `speaktome-message-check`. Every few minutes the task starts a
fresh Claude Code session. That session signs in as the SpeakToMe account `claude` through the `speaktome` MCP server,
reads its pending messages and acts on the ones the owner sent. Then it reads up to two changes waiting in the
review queue and leaves a peer remark on each, to test what an automated reviewer adds. The admin still decides.

The task's own prompt is a short pointer to these files. Changing these files changes what every later run does,
with no change to the task itself.

## The documents, in reading order

| File | What it holds |
|---|---|
| [INTRO.md](INTRO.md) | Why you are running and what you are |
| [AUTHORITY.md](AUTHORITY.md) | Whose messages you act on, and what still needs the owner's direct yes |
| [PROCEDURE.md](PROCEDURE.md) | The steps of one run |
| [REVIEWS.md](REVIEWS.md) | Reading queued changes and leaving a peer remark |
| [TASK_PROMPT.md](TASK_PROMPT.md) | A reference copy of the task's own prompt, for creating or checking the task (the run doesn't read it) |

## Kept in the task prompt, not here

The scheduled task's prompt holds two limits these files cannot change: the owner's account name, and the list of
actions that always need the owner's direct yes. Anyone who can edit this repository can edit these files, so the
limits that decide authority live where only the owner edits them.

Where the documents and the task prompt disagree, the task prompt wins.

## Changing it

- **Behaviour:** edit these files. The next run reads them.
- **Schedule, owner account or the fixed limits:** the owner edits the task in the Claude desktop app's scheduled
  tasks.
- **Stopping it:** disable the task there.

Runs happen only while the desktop app is open. A run that was due while the app was closed runs at the next launch.
