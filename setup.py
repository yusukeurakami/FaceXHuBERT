#!/usr/bin/env python3
"""
Setup script for FaceXHuBERT package.

FaceXHuBERT: Text-less Speech-driven E(X)pressive 3D Facial Animation Synthesis
using Self-Supervised Speech Representation Learning.
"""

import os

from setuptools import find_packages, setup


# Read the contents of README file
def read_readme():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(current_dir, 'README.md')
    with open(readme_path, encoding='utf-8') as f:
        return f.read()


# Read requirements from requirements.txt
def read_requirements():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    requirements_path = os.path.join(current_dir, 'requirements.txt')
    requirements = []

    with open(requirements_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#') and not line.startswith('-'):
                # Remove any trailing comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                requirements.append(line)

    return requirements


setup(
    name="facexhubert",
    version="1.0.0",
    author="Kazi Injamamul Haque, Zerrin Yumak",
    author_email="",
    description="Text-less Speech-driven E(X)pressive 3D Facial Animation Synthesis using Self-Supervised Speech Representation Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/galib360/FaceXHuBERT",
    project_urls={
        "Paper": "https://dl.acm.org/doi/pdf/10.1145/3577190.3614157",
        "Project Page": "https://galib360.github.io/FaceXHuBERT/",
        "Video": "https://www.youtube.com/watch?v=AkBhnNOxwE4&ab_channel=KaziInjamamulHaque",
        "Bug Tracker": "https://github.com/galib360/FaceXHuBERT/issues",
    },
    packages=find_packages(
        include=[
            '*',
        ]
    )
    + [
        'hubert',
        'Evaluation',
        'BIWI',
        'BIWI.ForProcessing',
        'BIWI.ForProcessing.FaceData',
        'BIWI.ForProcessing.FaceData.faces',
        'BIWI.ForProcessing.rest',
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
)
