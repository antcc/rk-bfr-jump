from pathlib import Path

from setuptools import find_packages, setup

# Read requirements from requirements.txt
requirements = Path("requirements.txt").read_text().splitlines()

setup(
    name="rkbfr_jump",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)
