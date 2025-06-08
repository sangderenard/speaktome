# Epoch Date Policy

All files that previously used `YYYY-MM-DD` dates now store timestamps as **Unix epoch seconds**. This applies to experience reports, user profiles, message filenames, and any new utilities.

Using epoch times ensures chronological ordering independent of timezone and avoids ambiguity across platforms. Example filename:

```
1720123456_v1_New_User_Experience_Simulation.md
```

Existing documents remain unchanged for historical accuracy, but new contributions should follow this convention. Update any scripts that generate dated filenames to emit epoch timestamps.
