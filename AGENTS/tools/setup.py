#!/usr/bin/env python3
"""Setup script for SPEAKTOME agent tools."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from setuptools import setup, find_packages
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

setup(
    name='speaktome-agent-tools',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "tomli; python_version<'3.11'",
        "importlib-metadata; python_version<'3.8'",
        "pytz",
        "ntplib",
        "pytest>=8.0",
    ],
)
