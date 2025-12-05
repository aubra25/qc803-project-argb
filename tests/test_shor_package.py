from shor_code_package.shor_code import ShorCode
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit import QuantumCircuit
import numpy as np

def test_encode_0_returns_expected_logical_state():
    #arrange
    encoder = ShorCode().encoder()
    ghz = Statevector([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)])
    target = ghz.copy() ^ ghz.copy() ^ ghz.copy() #Tensor three copies
    
    #act
    final_state = Statevector.from_instruction(encoder)

    #assert
    assert np.isclose(state_fidelity(final_state, target), 1)

def test_encode_1_returns_expected_logical_state():
    #arrange
    ghz_minus = Statevector([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, -1/np.sqrt(2)])
    target = ghz_minus.copy() ^ ghz_minus.copy() ^ ghz_minus.copy() #Tensor three copies
    
    #Ensure a |1> is the input to the encoder
    qc = QuantumCircuit(9)
    qc.x(0)
    qc.compose(ShorCode().encoder(), inplace=True)

    #act
    final_state = Statevector.from_instruction(qc)

    #assert
    assert np.isclose(state_fidelity(final_state, target), 1)
