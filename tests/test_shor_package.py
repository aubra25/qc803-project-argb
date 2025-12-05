from shor_code_package.shor_code import ShorQubit
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, AncillaRegister
from qiskit_aer import AerSimulator
import numpy as np
import pytest

class TestShorQubit():
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

    aer = AerSimulator()
    def aer_simulation(self, qc, shots = 10):
        result = self.aer.run(qc.decompose(), shots = shots).result()
        return result.get_counts()

    ##########################################
    # Tests
    ##########################################
    #Assignment 1
    def test_encode_0_returns_0_logical_state(self):
        #Arrange
        encoder = ShorQubit().encoder()
        target = self.logical_0()
        
        #Act
        final_state = qi.Statevector.from_instruction(encoder)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    def test_encode_1_returns_1_logical_state(self):
        #Arrange
        target = self.logical_1()
        
        #Ensure a |1> is the input to the encoder
        qc = self.ket_1_circuit().compose(ShorQubit().encoder())

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    #Assignment 2
    @pytest.mark.parametrize("stabilizer", ShorQubit().get_stabilizers())
    def test_logical_0_is_unaffected_by_stabilisers(self, stabilizer):
        #Arrange 
        sq = ShorQubit()
        qc = sq.encoder().compose(stabilizer)
        target = self.logical_0()

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    @pytest.mark.parametrize("stabilizer", ShorQubit().get_stabilizers())
    def test_logical_1_is_unaffected_by_stabilisers(self, stabilizer):
        #Arrange 
        sq = ShorQubit()
        qc = self.ket_1_circuit().compose(sq.encoder().compose(stabilizer))
        target = self.logical_1()

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    #Assignement 3
    @pytest.mark.parametrize("qubit, syndrome", 
                             [
                                 (0,"00100000"),
                                 (1,"00110000"),
                                 (2,"00010000"),
                                 (3,"00001000"),
                                 (4,"00001100"),
                                 (5,"00000100"),
                                 (6,"00000010"),
                                 (7,"00000011"),
                                 (8,"00000001"),
                              ])
    def test_bit_flip_gives_unique_syndrome(self, qubit, syndrome):
        #Arrange
        sq = ShorQubit()
        qc = sq.encoder()
        qc.add_register(AncillaRegister(8))
        qc.x(qubit)
        qc.compose(sq.stabilizer_measurement_circuit(), inplace=True)

        #Act
        result = self.aer_simulation(qc)

        #Assert
        #There should only be a single syndrome as the simulation is noiseless
        assert len(result.keys()) == 1
        assert list(result.keys())[0] == syndrome[::-1] #Bit order is reversed in qiskit

    @pytest.mark.parametrize("qubit, syndrome", 
                            [
                                (0,"10000000"),
                                (1,"10000000"),
                                (2,"10000000"),
                                (3,"11000000"),
                                (4,"11000000"),
                                (5,"11000000"),
                                (6,"01000000"),
                                (7,"01000000"),
                                (8,"01000000"),
                            ])
    def test_phase_flip_gives_degenerate_syndrome(self, qubit, syndrome):
        #Arrange
        sq = ShorQubit()
        qc = sq.encoder()
        qc.add_register(AncillaRegister(8))
        qc.z(qubit)
        qc.compose(sq.stabilizer_measurement_circuit(), inplace=True)

        #Act
        result = self.aer_simulation(qc)

        #Assert
        #There should only be a single syndrome as the simulation is noiseless
        assert len(result.keys()) == 1
        assert list(result.keys())[0] == syndrome[::-1] #Bit order is reversed in qiskit

