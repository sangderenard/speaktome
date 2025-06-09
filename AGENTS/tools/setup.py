from setuptools import setup, find_packages

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
