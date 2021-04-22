import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dspipeline",
    version="0.0.4",
    author="DF",
    author_email="",
    description="Data Science Ultimate Pipeline",
    long_description=long_description,
    url="https://github.com/chrisferreyra13/dspipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
