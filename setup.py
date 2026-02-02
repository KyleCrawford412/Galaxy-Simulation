"""Setup script for galaxy-sim package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="galaxy-sim",
    version="0.1.0",
    description="A senior-level N-body galaxy simulation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/galaxy-sim",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "gpu": [
            "jax>=0.3.0",
            "jaxlib>=0.3.0",
            "torch>=1.10.0",
            "cupy>=10.0.0",
        ],
        "export": [
            "imageio>=2.9.0",
            "opencv-python>=4.5.0",
        ],
        "config": [
            "pyyaml>=5.4.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "galaxy-sim=galaxy_sim.cli.main:main",
            "galaxy-sim-gui=galaxy_sim.ui.main:run_gui",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
