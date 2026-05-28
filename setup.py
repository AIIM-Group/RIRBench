from setuptools import setup, find_packages

setup(
    name="rirbench",
    version="0.1.0",
    description="Multi-metric evaluation framework for blind RIR generation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "soundfile>=0.10.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
    ],
)
