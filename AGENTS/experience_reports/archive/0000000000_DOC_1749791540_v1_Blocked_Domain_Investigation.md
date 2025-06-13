# Blocked Domain Investigation

**Date/Version:** 1749791540 v1
**Title:** Network requests during environment setup

## Overview
While running `setup_env.sh` the environment attempted to resolve packages from several domains that were blocked by the container's egress policy. The blocked domains were `api.nuget.org`, `sqlalche.me`, and `files.pythonhosted.org`.

## Prompts
- "I need a report on what package it was that those domains held and what conditions led to their being requested"

## Steps Taken
1. Executed `bash setup_env.sh -NoVenv -Codebases speaktome -Groups dev` to initialise the environment.
2. Observed failing network requests in the console output and logs.
3. Examined `pyproject.toml` to identify dependencies that might trigger those domains.
4. Searched the repository for direct references to the domains but found none.

## Observed Behaviour
- `poetry install` attempted to download dependencies such as `iniconfig`, `pluggy`, `pygments`, `numpy`, `pillow`, and others from `files.pythonhosted.org`.
- During installation, the tool also invoked build steps that attempted to contact `api.nuget.org` and `sqlalche.me`.
- The installation ultimately failed due to missing packages and PEP517 build errors.

## Analysis
- `files.pythonhosted.org` is the primary host for Python packages on PyPI. The environment attempted to fetch standard wheels from this domain for all dependencies listed in `pyproject.toml`.
- `api.nuget.org` hosts NuGet packages used by certain Python packages that rely on .NET or Windows tooling. This request likely originated from a build step in one of the dependencies—possibly `numpy` or `pillow`—that seeks prebuilt binaries or metadata via the NuGet API.
- `sqlalche.me` is the official home for SQLAlchemy. Even though SQLAlchemy is not an explicit dependency, some package metadata references this domain. When `pip` reads project metadata, it may attempt to resolve URLs listed under `home-page` or `project-url`, leading to this outbound request.

## Lessons Learned
- Running `setup_env.sh` without network access leads to repeated attempts to download packages from PyPI and related domains.
- Some packages reference external domains in metadata, which the installer tries to fetch even when the package itself is not required.

## Next Steps
- Consider configuring a local package mirror or caching proxy for offline setup.
- Document network requirements explicitly in `ENV_SETUP_OPTIONS.md` so future agents know when outbound access is necessary.
