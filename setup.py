from setuptools import find_packages, setup

with open("README.md") as read_file:
    long_description = read_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="snorkel-superglue",
    version="0.1.0",
    url="https://github.com/HazyResearch/snorkel-superglue",
    description="Applying snorkel (stanford.snorkel.edu) to superglue",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    keywords="machine-learning ai information-extraction weak-supervision",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    project_urls={
        "Homepage": "https://hazyresearch.github.io/snorkel-superglue/",
        "Source": "https://github.com/HazyResearch/snorkel-superglue/",
        "Bug Reports": "https://github.com/HazyResearch/snorkel-superglue/issues",
    },
)
