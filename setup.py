from setuptools import setup
import pathlib
import re

here = pathlib.Path(__file__).parent.resolve()
readme = (here / "README.md").read_text(encoding="utf-8")
version = re.search(
    '__version__ = "([^"]+)"',
    (here / "balaur/__init__.py").read_text(encoding="utf-8")
).group(1)


setup(
    name="balaur",
    version=version,
    author="mirandrom",
    description="Balaur: Language Model Pretraining with Lexical Semantic Relations",
    long_description_content_type="text/markdown",
    long_description=readme,
    url="https://github.com/mirandrom/balaur",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['balaur'],
    python_requires=">=3.10",
    install_requires=[
        "torch == 1.12.*",
        "pytorch-lightning == 1.7.*",
        "torchmetrics == 0.11.*",
        "transformers == 4.23.*",
        "datasets == 2.2.*",
        "apache-beam == 2.51.*",
        "wandb == 0.13.*",
        "nltk == 3.4",
        "inflect ~= 6.0.4",
        "plotly ~= 5.15.0",
        ],
)

