from setuptools import setup, find_packages

setup(
    name="hcm-transcription",
    version="0.1.0",
    description="Adaptive windowing for automated symbolic transcription of Hindustani vocals",
    author="Rhythm Jain, Claire Arthur",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "mirdata>=1.0.0",
        "librosa>=0.10",
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "python-Levenshtein",
        "tqdm",
    ],
)
