from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-PROJECT-02",
    version="0.1.0",
    author="Sankar Ayinala",
    description="A sample MLOps project 2",
    packages=find_packages(where="src"),        # ← This is the key change
    package_dir={"": "src"},                    # ← This tells it where the packages live
    install_requires=requirements,
)