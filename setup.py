"""
Setup configuration for the 'graph_viz' package.
"""
from setuptools import setup, find_packages

setup(
    name="graph_viz",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "numpy",
        "matplotlib",
        "Pillow",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "graph_viz = your_package.graph_animation:main",
        ],
    },
    author="Florian Richter",
    author_email="florian2richter@gmail.com",
    description="short package for generating colored 3-D plot",
    url="https://github.com/Florian2Richter/graph_viz",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
