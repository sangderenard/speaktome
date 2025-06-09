# Remove PureTensor Wrapper and Add Backend Guidelines

## Prompt History
```
check the agent messages outbox for an untitled markdown, if it's there in your version of the repo you need to read it and apply the comment in it to all the back ends and take out your dummy class pure tensor that nobody wants. also if you ran the tests from the venv torch would be there
```

## Summary
- Removed the `PureTensor` helper from `pure_backend.py` and restored the original list-based implementation.
- Added the backend implementation guidelines comment to every backend module.

