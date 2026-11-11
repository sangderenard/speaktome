# AUDIT: a stored GitHub access token was public; file and passphrase removed

**Date:** 1794372600 (2026-11-10)
**Title:** Stored access token in a public repository

## Scope
`AGENTS/github_pat.enc`, the four archived reports that describe it, and `AGENTS/tools/xor_cipher.py`.

## Methodology
Read-only: file size and type, git history, remote branches, repository visibility, and the reports. The file was
never decoded and the token was never used.

## Detailed Observations
- The file held a GitHub personal access token, added 2025-06-15. Its hidden text is 93 characters, the length of a
  fine-grained token. A report of the same day records a successful push made with it.
- The repository is public and the file is on about fifty remote branches.
- The protection was a repeating XOR with a passphrase, base64 encoded (`AGENTS/tools/xor_cipher.py`). The passphrase
  was written in plain text in four archived reports in this repository, one of which states it was not stored here.
  Tokens of that kind also begin with eleven known characters, which gives away most of a short key regardless.
- So the token has been readable by anyone since June 2025.

## Analysis
Removing the file from the current tree does not remove it from history or from the old remote branches, and history
cannot be made safe by rewriting it once a repository has been public. The only action that removes the risk is
revoking the token at GitHub. Whether it is still valid was not tested.

## Recommendations
1. The owner revokes the token at GitHub (Settings, Developer settings, Personal access tokens, Fine-grained tokens),
   after noting what it could reach.
2. No credential is ever stored in this repository again, encoded or not. `.gitignore` now names the file.
3. `xor_cipher.py` is kept only as history; it is an encoding, not encryption, and must not be used for secrets.
4. All identity and credential mechanics in this repository from before 2026-11-10 are historical evidence only. The
   service's SQL accounts are the one authority for who a person is.

## Prompt History
Owner: "investigate the possible token"

Owner: "okay so on the ambiguity, everything is old and is now in the light of a public exposure. the only thing that
is real users is the meowdoku (temporary name) sql that's the new everything"

Owner: "speaktome can be the name again and development can move there, all ... its user mechanics are now just
historical evidence. what is preventing you from scrubbing that token"
