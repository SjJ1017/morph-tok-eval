from pathlib import Path
from setuptools import setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

requirements_path = ROOT / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="morph-tok-eval",
    version="0.1.0",
    description="Morphological tokenizer evaluation utilities",
    long_description=README,
    long_description_content_type="text/markdown",
    author="morph-tok-eval contributors",
    license="MIT",
    py_modules=[
        "align",
        "data",
        "dynamic_program_segment",
        "get_vocabulary",
        "plot",
        "pos_correlation",
        "pos_tagger",
    ],
    packages=["morph_tok_eval"],
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
)
