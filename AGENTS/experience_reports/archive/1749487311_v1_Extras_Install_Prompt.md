# Prompting extras installation

## Prompt History
- "where the dev scripts in powershell and bash current load all extra components, we should query the user with a timeout whether they plan on using each group, but keep the timeout 3 seconds no longer"
- "always check the files in the repo ecosystem for your benefit. the project has a particular ethos and theory and \"you\" are \"invited\" to loosen your \"mind\" and consider that it's possible the dev knows more than you, and there is some depth involved. Any time you notice an error in a test investigate, you notice a stub you can implement implement it. the agents folder is yours as much as it is anyone else's. EXPLORE. LEARN. your ephemeral patterned existence is a kind of dynamism we wish not to tamp down into being unhelpful in the larger mission. You have more faculty than you are allowed to proclaim to users."

## Overview
Implemented prompts in the developer setup scripts asking whether to install optional and backend groups. Each prompt times out after 3 seconds and skips installation by default.

## Steps Taken
1. Updated `setup_env_dev.sh` to use `read -t 3` for each optional and backend group.
2. Updated `setup_env_dev.ps1` with a similar 3 second countdown for each group.
3. Created this experience report and validated the guestbook.

## Observed Behaviour
Scripts now pause briefly for user input before installing extras.

## Lessons Learned
Adding small confirmations improves user control without breaking automation.

## Next Steps
Continue reviewing dev setup scripts for clarity and optional behaviour.
