from setuptools import find_packages, setup

setup(
    name="rkbfr_jump",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.5",
        "pandas==2.1.4",
        "scipy==1.11.4",
        "arviz==0.18.0",
        "matplotlib==3.8.2",
        "scikit-fda==0.9.1",
        "scikit-learn==1.4.0",
        "numba==0.58.1",
        "Eryn==1.1.7",  # If using a custom version of Eryn, comment out this line
    ],
)
