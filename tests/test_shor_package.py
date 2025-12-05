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
        return result

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

    #Assignement 3 and 4
    @pytest.mark.parametrize("qubit, syndrome", 
                             [
                                 (0,"00 10 00 00"),
                                 (1,"00 11 00 00"),
                                 (2,"00 01 00 00"),
                                 (3,"00 00 10 00"),
                                 (4,"00 00 11 00"),
                                 (5,"00 00 01 00"),
                                 (6,"00 00 00 10"),
                                 (7,"00 00 00 11"),
                                 (8,"00 00 00 01"),
                              ])
    def test_bit_flip_gives_unique_syndrome(self, qubit, syndrome):
        #Arrange
        sq = ShorQubit()
        qc = sq.encoder()
        qc.add_register(AncillaRegister(8))
        qc.x(qubit)
        qc.compose(sq.syndrome_correction_circuit(correct_syndromes=False), inplace=True)

        #Act
        result = self.aer_simulation(qc).get_counts()

        #Assert
        #There should only be a single syndrome as the simulation is noiseless
        assert len(result.keys()) == 1
        assert list(result.keys())[0] == syndrome[::-1] #Bit order is reversed in qiskit

    @pytest.mark.parametrize("qubit, syndrome", 
                            [
                                (0,"10 00 00 00"),
                                (1,"10 00 00 00"),
                                (2,"10 00 00 00"),
                                (3,"11 00 00 00"),
                                (4,"11 00 00 00"),
                                (5,"11 00 00 00"),
                                (6,"01 00 00 00"),
                                (7,"01 00 00 00"),
                                (8,"01 00 00 00"),
                            ])
    def test_phase_flip_gives_degenerate_syndrome(self, qubit, syndrome):
        #Arrange
        sq = ShorQubit()
        qc = sq.encoder()
        qc.add_register(AncillaRegister(8))
        qc.z(qubit)
        qc.compose(sq.syndrome_correction_circuit(correct_syndromes=False), inplace=True)

        #Act
        result = self.aer_simulation(qc).get_counts()

        #Assert
        #There should only be a single syndrome as the simulation is noiseless
        assert len(result.keys()) == 1
        assert list(result.keys())[0] == syndrome[::-1] #Bit order is reversed in qiskit

    #Assignment 5
    @pytest.mark.parametrize("qubit, pauli_error, target", [(qubit, pauli_error, target) 
                                                    for qubit in range(9) 
                                                    for pauli_error in ["X", "Z", "Y"] 
                                                    for target in [0, 1]])
    def test_single_pauli_errors_are_corrected(self, qubit, pauli_error, target):
        #Arrange
        sq = ShorQubit()
        qc = QuantumCircuit(9 + 1) #Nine physical qubits, one ancilla
        if target == 1:
            qc.x(0)
        qc.compose(sq.encoder(), inplace = True)

        match pauli_error:
            case "X":
                qc.x(qubit)
            case "Y":
                qc.y(qubit)
            case "Z":
                qc.z(qubit)
        
        qc.compose(sq.syndrome_correction_circuit(), inplace = True)
        qc.save_density_matrix(range(9), label="p")

        target = self.logical_0() if target == 0 else self.logical_1()

        #Act
        result = self.aer_simulation(qc, shots=1)
        final_statevector = result.data()["p"].to_statevector() #Assumes the final state is pure.

        #Assert
        assert len(result.get_counts().keys()) == 1
        assert np.isclose(qi.state_fidelity(final_statevector, target), 1)

    def test_multiple_bit_flips_are_uncorrectable(self):
        #Arrange
        sq = ShorQubit()
        qc = QuantumCircuit(9 + 1) #Nine physical qubits, one ancilla
        qc.compose(sq.encoder(), inplace = True)
        qc.x(0)
        qc.x(1)
        qc.compose(sq.syndrome_correction_circuit(), inplace = True)
        qc.save_density_matrix(range(9), label="p")

        target = self.logical_0()

        #Act
        result = self.aer_simulation(qc, shots=1)
        final_statevector = result.data()["p"].to_statevector() #Assumes the final state is pure.

        #Assert
        assert len(result.get_counts().keys()) == 1
        assert np.isclose(qi.state_fidelity(final_statevector, target), 1)


        



