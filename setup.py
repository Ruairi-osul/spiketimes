from setuptools import setup, find_packages

description = ""
long_description = ""
setup(
    name="siketimes",
    description=description,
    long_description=long_description,
    version="0.0.1a1",
    url="https://github.com/Ruairi-osul/spiketimes",
    author="Ruairi O'Sullivan",
    author_email="ruairi.osullivan.work@gmail.ie",
    license="GNU GPLv3",
    keywords="neuroscience science electrophysiology ephys",
    project_urls={"Source": "https://github.com/Ruairi-osul/spiketimes"},
    packages=find_packages(),
    python_requires=">=3.3",
)

