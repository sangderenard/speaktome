from setuptools import setup, find_packages

setup(
    name="Alpha",
    version="1.0.1",
    description="A simulation and monitoring program.",
    author="default uuid",
    packages=find_packages(include=["Alpha", "Alpha.*"]),
    install_requires=[
        "pybullet",
        "pygame",
        "torch",
        "torch-geometric",
        "pymunk",
    ],
    entry_points={
        'console_scripts': [
            'alpha=Alpha.main:main',  # Create a CLI command 'alpha'
        ],
    },
    python_requires='>=3.8',
)
