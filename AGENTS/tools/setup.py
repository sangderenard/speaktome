
from __future__ import annotations

try:
    from setuptools import setup, find_packages
except Exception:
    print(
        "\n"
        "+-----------------------------------------------------------------------+\n"
        "| Imports failed. Run setup_env or setup_env_dev and select every    |\n"
        "| project and module you plan to use. Missing packages mean setup was |\n"
        "| skipped or incomplete.                                             |\n"
        "+-----------------------------------------------------------------------+\n"
    )
    raise
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
