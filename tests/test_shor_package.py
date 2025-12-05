from shor_code_package.shor_code import ShorCode
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
import numpy as np
import pytest

class TestShorCode():
    ##########################################
    # Useful resources for use in tests
    ##########################################
    def ket_1_circuit(self):
        qc = QuantumCircuit(9)
        qc.x(0)
        return qc
    
    def logical_0(self):
        ghz = qi.Statevector([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)])
        return ghz ^ ghz ^ ghz
    
    def logical_1(self):
        ghz_minus = qi.Statevector([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, -1/np.sqrt(2)])
        return ghz_minus ^ ghz_minus ^ ghz_minus

    ##########################################
    # Tests
    ##########################################
    def test_encode_0_returns_0_logical_state(self):
        #Arrange
        encoder = ShorCode().encoder()
        target = self.logical_0()
        
        #Act
        final_state = qi.Statevector.from_instruction(encoder)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    def test_encode_1_returns_1_logical_state(self):
        #Arrange
        target = self.logical_1()
        
        #Ensure a |1> is the input to the encoder
        qc = self.ket_1_circuit().compose(ShorCode().encoder())

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    @pytest.mark.parametrize("stabilizer", ShorCode().get_stabilizers())
    def test_logical_0_is_unaffected_by_stabilisers(self, stabilizer):
        #Arrange 
        sc = ShorCode()
        qc = sc.encoder().compose(stabilizer)
        target = self.logical_0()

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    @pytest.mark.parametrize("stabilizer", ShorCode().get_stabilizers())
    def test_logical_1_is_unaffected_by_stabilisers(self, stabilizer):
        #Arrange 
        sc = ShorCode()
        qc = self.ket_1_circuit().compose(sc.encoder().compose(stabilizer))
        target = self.logical_1()

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)
