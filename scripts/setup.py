from setuptools import setup, find_packages

setup(
    name="pyenmeval",
    version="0.1.4",
    author="Diego Gómez",
    author_email="gomezdiego1902@gmail.com",
    description="A Python implementation of ENMeval for ecological niche model evaluation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ddk1902/PyENMeval_V1.0",
    project_urls={
        "Bug Tracker": "https://github.com/ddk1902/PyENMeval_V1.0/issues",
        "Source Code": "https://github.com/ddk1902/PyENMeval_V1.0",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
)

