from shor_code_package.shor_code import ShorQubit, ConcatenatedShorQubit
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, AncillaRegister
from qiskit_aer import AerSimulator
import numpy as np
import pytest

class TestUtilities:
    """
    Class holding static methods for useful in testing.
    """
    def ket_1_circuit():
        qc = QuantumCircuit(9)
        qc.x(0)
        return qc
    
    def logical_0():
        ghz = qi.Statevector([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)])
        return ghz ^ ghz ^ ghz
    
    def logical_1():
        ghz_minus = qi.Statevector([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, -1/np.sqrt(2)])
        return ghz_minus ^ ghz_minus ^ ghz_minus

    def aer_simulation(qc, shots = 5):
        aer = AerSimulator()
        result = aer.run(qc.decompose(), shots = shots).result()
        return result
    
@pytest.mark.skip()
class TestShorQubit():
    #Assignment 1
    def test_encode_0_returns_0_logical_state(self):
        #Arrange
        encoder = ShorQubit().encoder()
        target = TestUtilities.logical_0()
        
        #Act
        final_state = qi.Statevector.from_instruction(encoder)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    def test_encode_1_returns_1_logical_state(self):
        #Arrange
        target = TestUtilities.logical_1()
        
        #Ensure a |1> is the input to the encoder
        qc = TestUtilities.ket_1_circuit().compose(ShorQubit().encoder())

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
        target = TestUtilities.logical_0()

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    @pytest.mark.parametrize("stabilizer", ShorQubit().get_stabilizers())
    def test_logical_1_is_unaffected_by_stabilisers(self, stabilizer):
        #Arrange 
        sq = ShorQubit()
        qc = TestUtilities.ket_1_circuit().compose(sq.encoder().compose(stabilizer))
        target = TestUtilities.logical_1()

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
        result = TestUtilities.aer_simulation(qc).get_counts()

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
        result = TestUtilities.aer_simulation(qc).get_counts()

        #Assert
        #There should only be a single syndrome as the simulation is noiseless
        assert len(result.keys()) == 1
        assert list(result.keys())[0] == syndrome[::-1] #Bit order is reversed in qiskit

    #Assignment 5
    @pytest.mark.parametrize("qubit, pauli_error, target", [(qubit, pauli_error, target) 
                                                    for qubit in range(9) 
                                                    for pauli_error in ["Y"] 
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

        target = TestUtilities.logical_0() if target == 0 else TestUtilities.logical_1()

        #Act
        result = TestUtilities.aer_simulation(qc, shots=1)
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

        target = TestUtilities.logical_0()

        #Act
        result = TestUtilities.aer_simulation(qc, shots=1)
        final_statevector = result.data()["p"].to_statevector() #Assumes the final state is pure.

        #Assert
        assert len(result.get_counts().keys()) == 1
        assert np.isclose(qi.state_fidelity(final_statevector, target), 1)

class TestConcanetatedShorQubit():

    @pytest.mark.parametrize("input_state", [0, 1])
    def test_encoder_produces_logical_states(self, input_state):
        #Arrange
        #Construct target stabilizer state. The stabilizers are like the ones for the ordinary Shor code for each
        #of the 9 qubit groupings plus an addtional set of stabilizers which are translations of the originals of
        #using the logical operators of the inner code.
        shor_code_stabilizers = ["XXXXXXIII", "IIIXXXXXX", "ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ"]
        zl = "X"*9
        xl = "Z"*9
        il = "I"*9
        zll_stabilizer = "-"+9*xl if input_state == 1 else 9*xl #This stabilizer distinguishes between logical 0 and logical 1.
        translator = str.maketrans({"X": xl, "Z": zl, "I": il})
        concatenated_code_stabilizer = [scs.translate(translator) for scs in shor_code_stabilizers]
        inner_code_stabilizers = [n * il + scs + (8-n) * il for n in range(9) for scs in shor_code_stabilizers]
        
        target = qi.StabilizerState.from_stabilizer_list([zll_stabilizer, *concatenated_code_stabilizer, *inner_code_stabilizers], allow_redundant=True)
        
        #Construct encoding circtui
        qc = QuantumCircuit(9**2)
        if input_state == 1:
            qc.x(0) #input |1>
        csq = ConcatenatedShorQubit(2)
        qc.compose(csq.encoder(), inplace=True)

        #Act
        encoded_logical_0 = qi.StabilizerState(qc)

        #Assert
        assert encoded_logical_0.equiv(target)