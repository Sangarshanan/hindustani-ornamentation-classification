from setuptools import setup, find_packages

setup(
    name="hcm-transcription",
    version="0.1.0",
    description="Hindustani Ornamentation Classification and Transcription",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "librosa==0.11.0",
        "numpy==2.4.3",
        "scipy==1.17.1",
        "pandas==3.0.1",
        "matplotlib==3.10.8",
        "seaborn==0.13.2",
        "scikit-learn==1.8.0",
        "pyyaml==6.0.3",
        "python-Levenshtein==0.27.3",
        "tqdm==4.67.3",
        "jupyterlab==4.5.6",
        "compiam==0.4.1"
    ],
)
