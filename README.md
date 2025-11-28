# QC 803: Final Project Repository Template

This is an optional template for the QC 803 final project based on https://packaging.python.org/en/latest/tutorials/packaging-projects/.

## File Structure
```
project_repo/
├── pyproject.toml
├── README.md
├── src/
│   └── example_package/
│       ├── __init__.py
│       └── example.py
└── tests/
    └── test_example.py
```

#### `pyrproject.tml`
This is a configuration file, which you can read more about at https://packaging.python.org/en/latest/tutorials/packaging-projects/. This is primarily for specifying your build system, project metadata, and dependencies.


#### `README.md` 
This document should clearly describe both the structure of your code and how to run it, including examples.

#### `src/`
This is the folder for your source code, which will include all of the packages that you build. For this project, you will likely just make a single package `src/project_name/` containing an `__init__.py` file and the files containing the functions, classes, etc. provided by your package.

#### `tests/`
This is the folder that will contain all of the unit tests that you write.

## Installation

Before installing, we recommend first setting up a virtual environment.
```sh
python -m venv .venv
source .venv/bin/activate
```

Then, you can install the project package using `pip` as shown below.
```sh
pip install .
```

## Tests
Unit tests can be run using `pytest` (see https://docs.pytest.org/en/stable/ for more details), which should output something like the following.
```
======================================== test session starts ========================================
platform darwin -- Python 3.14.0, pytest-9.0.1, pluggy-1.6.0
rootdir: /path/to/qc803-project-repo-template
configfile: pyproject.toml
collected 1 item                                                                                    

tests/test_example.py .                                                                       [100%]

========================================= 1 passed in 0.00s =========================================
```