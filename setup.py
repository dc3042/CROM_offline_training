from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    "h5py==3.4.0", "numpy>=1.22", "pynvml==11.0.0", "pytorch_lightning==1.6.5", "torch==1.11.0"]

setup(
    name="run_crom",
    version="1.0.0",
    author="David Cho",
    author_email="dc3042@columbia.edu",
    description="A package for training and testing CROM weights",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/dc3042/CROM_offline_training",
    packages=find_packages(),
    entry_points ={
            'console_scripts': [
                'run_crom = run_crom.run_crom:main'
            ]
        },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
