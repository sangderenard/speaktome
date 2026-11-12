# Continue native Studio resource ownership

Source/CPU checkpoint: `nodus/docs/NATIVE_STUDIO_STORAGE_2026-11-11.md` and
`nodus/out/native-field-storage/`. Existing protected link-only child destroys
its context/programs; new canonical memory backend is not live-integrated.

Next required work is persistent process-local context/resource ownership in
that existing protected-child/row lifecycle, exact scene byte handoff, all-path
producer/consumer accounting and separately authorized admission/verification.
Keep the worker execution hold and unverified continuation refusal until those
gates genuinely pass. No service reload, driver work or rendering is implied.
