from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="MLOps pipeline for tweet sentiment classification "
    "(Flask + DVC + MLflow + DagsHub).",
    author="rounak",
    license="",
)
