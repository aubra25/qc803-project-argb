# QC 803: Concatenated Shor codes

This project provides the means to construct circuits of concatenated Shor encoded logical qubits.
It consists of a single package, shor_code_package.

## Installation
1. Clone the repo.
2. Make sure you are using Python version 3.13.1 as Qiskit Aer is not compatible with the newest version of Python. Use for example pyenv to manage Python versions (https://github.com/pyenv/pyenv).
3. Navigate to the root of the repo.
4. Create a new python virtual environment: 
```sh
python -m venv .venv
source .venv/bin/activate
```
5. Install the dependencies
```sh
pip install .
```
You are now ready to run unit tests or use the package.

## Getting started
Check the getting_started.ipynb to hit the ground running! It shows basic usage and provides an example of creating a circuit for preparing an encoded GHZ state.

## File Structure
```
project_repo/
├── pyproject.toml
├── README.md
├── getting_started.ipynb
├── experiments.ipynb
├── src/
│   └── shor_code_package/
│       ├── __init__.py
│       └── shor_code.py
├── tests/
|   └── test_shor_package.py
├── data/
|   └── ...
└── figures/
    └── ...
```
### Notable files
#### getting_started.ipynb
Notebook describing basic usage of the classes in the project.

#### experiments.ipynb
Notebook containing code for experiments with the error correction capabilities of the concatenated Shor code. Also contains code for post processing simulation data and visualising the results based on saved binarys.

#### data/
Contains binarys containing the post processed simulation data used for the visualisation in the accompanying report. Overwritten when post processing simulation results in the experiments.ipynb.

#### figures/
The figures based on the data from the data folder.

## Structure of the code
The shor_code_package contains two classes. They are described in the following.
### ConcatenatedShorQubit
This class takes an integer, $n$, indicating the number of times the Shor code has been concatenated with itself, and provides methods for constructing the encoder, error correction circuit, and logical gates of the encoding. $n=0$ indicates no encoding, $n=1$ is the ordinary Shor code and $n=2$ is the Shor code concatenated with itself etc.

The encoder, logical H, S, X and Z are constructed recursively. The methods have a base case and inductive steps. Essentially one can construct the gate for $n + 1$ by using the logical gates of a ConcatenatedShorQubit characterised by $n$. An option is available to interleave logical operations with error correction.

The error correction circuit (syndrome_correction_circuit method) was not implemented using the recursive approach due to how classical registers behave in Qiskit. Instead all stabilizers and recovering circuits are collected and classical conditioning is specified in a more imperative manner.

### ShorCircuit
This class is responsible for constructing QuantumCircuits of qubits encoded in the concatenated Shor code. It stores the indices of the physical qubits and makes composition of gates easy. Error corrections is inserted as placeholders since their classical conditioning parts make them hard to simply handle as Qiskit Instructions. The placeholders are replaced with the actual error correction circuits when the quantum circuit is exported using the get_circuit method in the end (see getting_started.ipynb for example).

The ShorCircuit class also implements the controlled not gate. It works between logical qubits of any encoding level. If the encoding level is the same, CX is implemented transversally. If encoding levels mean more target than control physical qubits, one control qubit simply targets more target qubits. If encoding levels result in more control than target qubits then several control qubits target each target qubit.
Since the encoder for the ConcatenatedShorQubit includes a Hadamard gate, logical CX gates are not necessarily implemented using physical CX gates. Depending on the parity of the two encoding levels the actual two qubit gates acting on the physical qubits may be CX, CZ or CX or CZ conjugated with Hadamards on the control line.

## Tests
### Purpose of each test
In test_shor_package.py there is a comprehensive test suite. A doc string at the beginning of each test method explains the purpose of the test.

### Running the tests
Unit tests can be run using `pytest` which should output something like the following.
```
=================================================== test session starts ===================================================
platform darwin -- Python 3.13.10, pytest-9.0.1, pluggy-1.6.0
rootdir: /path/to/qc803-project-argb
configfile: pyproject.toml
collected 184 items                                                                                                       

tests/test_shor_package.py ........................................................................................ [ 47%]
................................................................................................                    [100%]

============================================= 184 passed in 150.22s (0:02:30) =============================================
```