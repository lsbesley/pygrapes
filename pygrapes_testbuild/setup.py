from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Read the README for long description
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="pygrapes",
    version="0.1.0",
    author="Luke Besley",
    author_email="luke.s.besley@gmail.com",
    description="PyTorch-based framework for 3D grazing incidence ptychography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lsbesley/pygrapes",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "h5py",
        "scikit-image",
        "scipy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
