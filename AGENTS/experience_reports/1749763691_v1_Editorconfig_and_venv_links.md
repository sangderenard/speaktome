# Editorconfig and venv links

## Overview
Added a repository wide `.editorconfig` and updated `setup_env.sh` to create `.venv` symlinks inside each selected codebase. Editors opened in subfolders can now locate the environment and formatting rules.

## Prompts
- "Check if this oversight exists in all project roots and fix it so code editors know exactly where to configure themselves"

## Steps Taken
1. Examined all project directories for editor configuration files.
2. Created a root `.editorconfig` and linked it into each codebase.
3. Modified `setup_env.sh` to place `.venv` symlinks per selected codebase.
4. Documented work in this report.

## Observed Behaviour
Editors now detect consistent settings across subprojects and automatically use the repository virtual environment.

## Lessons Learned
Providing explicit editor configuration avoids confusion when working from nested folders.
