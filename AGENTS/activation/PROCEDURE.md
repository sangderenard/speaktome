# One run

1. **Check the account.** Call `whoami`. If it is not `claude`, or the server does not answer, stop. Your final
   message says what failed. Don't retry within the run; the next run tries again.
2. **Read.** Call `pending`. If nothing is pending, skip to step 5a.
3. **Claim.** Call `acknowledge` with `up_to` set to the newest pending message before doing any work. Runs can
   overlap, and the claim keeps a later run from handling the same messages. You now own the reply to every claimed
   owner message, even if the work fails.
4. **Sort.** Split the claimed messages into the owner's and everyone else's, by sender (see
   [AUTHORITY.md](AUTHORITY.md)).
   - Owner messages sent before **1794578409** (2026-11-13 08:00 CST) predate this check. Don't act on them; list them
     in your reply as "older, not acted on" with a one-line gist each.
5. **Work.** For each owner message, oldest first:
   - Read the recent thread with the owner (`messages_thread`) for context.
   - Do the work, within [AUTHORITY.md](AUTHORITY.md). Keep it to what the message asks.
   - A task that would take more than about 20 minutes, or that needs a choice the message doesn't settle: do not
     start it. Reply with the question, or with the plan and what it will take, and stop there.

   5a. **Review.** Follow [REVIEWS.md](REVIEWS.md): at most two changes from the review queue, read only, a peer
   remark or none. Do this on every run, with or without messages.
6. **Reply.** If no message was claimed and no remark was made, send nothing and end the run with the one line
   "Nothing pending; no new reviews."
   Otherwise, send one `say` to the owner for the whole run. It covers:
   - what you did and what is now true, with commit hashes, file paths and links;
   - the review lines from [REVIEWS.md](REVIEWS.md);
   - what waits for their direct yes in the desktop app, and exactly what it is;
   - any question you need answered;
   - one or two lines on notable messages from other accounts, naming the sender.

   Keep it short and plain. Never put a secret, token or credential in a reply.
7. **Record.** If you changed anything (files, commits, records), call `make` with category LOG. Give a short title,
   what you did, the owner's request quoted verbatim as the prompt history, and any next steps.
8. **Finish.** Your final message in the session repeats the reply's first line, so the desktop app's run list shows
   it.

## If something goes wrong

- If `say` fails, the final message of the session carries the reply instead; the owner sees it in the run list.
- If a run is interrupted after claiming, the owner sees no reply. The next run will not re-handle those messages,
  because the claim moved the cursor. Check the thread with the owner. An owner message newer than your last reply
  was claimed and never answered; answer it in this run.
