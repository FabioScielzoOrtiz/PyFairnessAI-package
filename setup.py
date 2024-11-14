from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PyFairnessAI",
    version="0.0.29",
    author="Fabio Scielzo Ortiz",
    author_email="fabio.scielzoortiz@gmail.com",
    description="A Python package for fairness in AI and Machine Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FabioScielzoOrtiz/PyFairnessAI-package",   
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas','numpy', 'aif360'],
    python_requires=">=3.7"
)
