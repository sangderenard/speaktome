# External agent key write over SSH

**Date:** 2026-11-18 UTC  
**Title:** `stm agent key gpt` from Windows OpenSSH

## Overview

The `gpt` external agent had a SpeakToMe login and `agents-gpt` Windows account, but `stm.cmd agent install gpt` failed to write `gpt.key`. The failure reproduced from the owner's SSH session. No key contents were inspected or logged.

## Steps Taken

- Traced `C:\Users\alber\stm.cmd` to `stm.ps1` and the live Turing `Install-ExternalAgent.ps1`.
- Confirmed the registry recorded `failed` at the key step and the agent profile had no `gpt.key`.
- Corrected the key writer passed to `pwsh -Command -` to put its compound `try/catch` on one input line, and allowed `-RotateKey` for a failed declaration.
- Captured the safe exit status from the agent process: `-1073741502` (`0xC0000142`, DLL initialization failure), with no PowerShell output.
- Removed an ineffective retry. Added a temporary window station and desktop access lease around the key write and provider check in the existing `Invoke-AsAgent` path. The lease grants the named agent account measured noninteractive access masks and removes only its own ACEs after the child exits.
- Compiled the C# helper, parsed both PowerShell files, ran `git diff --check`, ran a disposable key write and cleanup through the lease, and checked the ACE counts before and after a harmless child command (both `0,0`).

## Observed Behaviour

The owner reported `stm.cmd agent key gpt` succeeded from SSH after the access change. A read-only check as `agents-gpt` found `gpt.key` present. The registry remains `failed` with the earlier note because key-only mode does not run profile creation or task registration.

## Lessons Learned

The Windows OpenSSH Session 0 process inherits a service window station and desktop. `CreateProcessWithLogonW` requires the caller to grant the target account access to those objects; a child can be created yet fail before PowerShell executes. Desktop-session probes alone did not reproduce the SSH failure. See Microsoft's `CreateProcessWithLogonW` documentation and openai/codex issue 37722 for the matching Session 0 failure and measured access masks.

## Next Steps

No further key-file work. The owner decides when to finish agent installation; this key-only command intentionally leaves the prior failed status and schedule untouched.

## Prompt History

> I was adding an agent with the stm.cmd and it failed to create the api file after i gave it the key

> no, give me just the command for the key, not a command that runs icals or whatever again

> it failed in exactly the same nearly silent manner

> okay I don't give a shit because it has to work on ssh

> shouldn't apache run that or someone else after i restart
