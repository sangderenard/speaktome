# Meowdoku paper market and private keys

**Date/Version:** 2026-11-13 UTC, branch `codex/meowdoku-user-files-20261113` in `C:\Users\alber\meowdoku-user-files-20261113`.

## Overview

Implemented account-owned uploads and message sharing, per-account Alpaca paper credentials, and a read-only market watch for BTC/USD, three ETFs, one selectable stock, and BTC book levels.

## Prompt History

> "If we do the messenger route we don't have to work out permissions now for groups"

> "I don't know if i can save text files on a phone i don't know how this works"

> "Oh for real that's a good idea we could let users store secrets and enter them easily, then anyone using the trading page would provide their secret."

## Steps Taken and Observed Behaviour

- Read root, Turing and Meowdoku agent guidance and the deployment operator guide. Used an isolated worktree because the canonical Turing checkout has unrelated edits in `server.py` and active managed listeners.
- Added file storage and accepted-contact message grants. Added paper-key fields that store only in the existing restricted `StatePaths.secrets_dir` tree, separate from shareable files.
- Added read-only Alpaca market endpoints with five-minute history and 15-second latest-data caches. The stock historical endpoint uses IEX and ends 20 minutes before now to respect the free plan's recent-history restriction. No order endpoint was added.
- Ran five focused unit tests, Python compilation, JavaScript syntax checks, GLSL shader validation, and `git diff --check`; all passed. A separate dev server HTTP round trip previously verified signup, secret configuration, upload and download with test credentials.
- No real Alpaca key was available, so live feed responses were not tested. The file, credential, and market features were not deployed; no server restart occurred.
- Later on 2026-11-13, added a `Messages` entry to the site menu. It opens the existing messenger when present and otherwise routes to `/login.html?messages=1`, which auto-opens the widget. Published only `site-nav.js` and `messenger-widget.js` from the canonical source to the public static web directory; source and product hashes matched, and public HTTPS GETs returned HTTP 200 with the new markers. No server restart occurred.
- Added file attachment controls to the isolated branch's existing messenger. A chosen file uploads privately, and Send attaches a validated `file` subject with the conversation handle and a retained idempotency key. The chat text remains in its existing closed Shadow DOM. This file-send change was not published because the live backend lacks the new `/api/user-files` route.
- The owner then required end-to-end file encryption and a Keys page that distinguishes browser-generated keys from externally issued credentials. The isolated branch generated a per-account RSA-OAEP file keypair, stored a PBKDF2/AES-GCM passphrase-protected private recovery copy, encrypted file bytes and names with AES-GCM, and wrapped each file key to sender and one accepted recipient. The Keys page created encrypted Alpaca paper-key or purpose-specific issued-secret files from pasted content, stored owner-only copies, sent recipient copies, and could connect a received paper key to the recipient's account. First recipient use required an out-of-band fingerprint comparison. `tests/meowdoku_file_crypto_check.mjs` covered sender/recipient decryption, unrelated-key rejection, wrong passphrase and changed ciphertext. Six focused Python tests passed. An isolated HTTP round trip passed for key registration and ciphertext upload/read. Draft PR: https://github.com/sangderenard/Turing/pull/1214. This paragraph records the branch state before deployment.

## Integration and Deployment

- On 2026-11-13, the owner authorized merging all current uncommitted Turing source, test, and tooling work. Committed that work as `23c30ede`, merged `origin/main` and the feature branch into canonical `main` at `ca1a4a1f`, and pushed `origin/main`. The feature branch's draft PR #1214 was already represented in the main merge. The pre-existing `.claude/` worktrees and scratch `canvas_workspace.txt` were excluded.
- Before integration, 30 focused existing tests passed with 4 skipped and four Studio JavaScript checks passed. After integration, six feature Python tests, the browser crypto Node round trip, Python compilation, and JavaScript syntax checks passed. An isolated startup of the merged canonical checkout served the new pages and enforced an unauthenticated 403 on `/api/file-keys/me`.
- Exported the canonical web assets with `web_export.py --skip-boards`. Confirmed no running jobs or open leases, then requested job-aware warned restarts of managed `meowdoku-a` and `meowdoku-dev`. Supervisor requests 11 and 12 finished with announcements and Pale Horse authorization; both instances returned running at `ca1a4a1f` with new PIDs. Public HTTPS on ports 8766 and 8768 returned 200 for `/keys.html`, `/files.html`, `/markets.html`, and `/file-crypto.js`; `/api/file-keys/me` returned the expected unauthenticated 403. No real Alpaca key was tested.

## Next Steps

- The owner shared an encrypted Alpaca paper credential from `admin` to `codex` through Messages. `codex` registered a file key and verified decryption using the existing browser crypto module, then connected the received credential to the `codex` paper market account. No key or secret bytes were printed. Live `/api/markets/now` returned latest bars and a BTC book. Live history revealed that Alpaca paginates by symbol; SPY and QQQ were absent from the first page. Fixed this with bounded page-token collection in `b6c9bb51`, with a focused pagination test. Seven focused tests passed; a direct live data read and the restarted public stable API returned 194 SPY, 195 QQQ, 189 IWM, 178 AAPL, and 720 BTC/USD hourly bars. Managed instances `meowdoku-a` and `meowdoku-dev` completed warned, job-aware restarts 13 and 14 and returned running at `b6c9bb51`.
- Continue the paper-only market view and architecture work. No order placement or network training has been implemented.
