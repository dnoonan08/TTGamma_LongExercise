#!/usr/bin/env python
import sys
import os.path
from setuptools import (
    setup,
    find_packages,
)


about = {}
with open(os.path.join("ttgamma", "version.py")) as f:
    exec(f.read(), about)


needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup(name="ttgamma",
      version=about["__version__"],
      packages=find_packages(),
      include_package_data=True,
      description="An analysis package",
      long_description_content_type="text/markdown",
      url="https://github.com/dnoonan08/TTGamma_LongExercise",
      install_requires=[
          "coffea",
          "xxhash",  # for LZ4 input files
      ],
      setup_requires=["flake8"] + pytest_runner,
)
