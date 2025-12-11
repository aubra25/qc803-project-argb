from shor_code_package.shor_code import ShorQubit, ConcatenatedShorQubit, ShorCircuit
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, AncillaRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import HGate
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
    
    def logical_state_stabilizers(input_state, include_ancilla = False):
        #The stabilizers are like the ones for the ordinary Shor code for each
        #of the 9 qubit groupings plus an addtional set of stabilizers which are translations of the originals of
        #using the logical operators of the inner code.
        shor_code_stabilizers = ["XXXXXXIII", "IIIXXXXXX", "ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ"]
        zl = "X"*9
        xl = "Z"*9
        il = "I"*9
        zll = 9*xl if input_state == 1 else 9*xl #This stabilizer distinguishes between logical 0 and logical 1.
        translator = str.maketrans({"X": xl, "Z": zl, "I": il})
        concatenated_code_stabilizer = [scs.translate(translator) for scs in shor_code_stabilizers]
        inner_code_stabilizers = [n * il + scs + (8-n) * il for n in range(9) for scs in shor_code_stabilizers]
        
        stabilizers = [*concatenated_code_stabilizer, *inner_code_stabilizers]
        if include_ancilla:
            stabilizers = ["I" + s for s in stabilizers] #Adding I for the ancilla.
            stabilizers.append("Z" + 9**2*"I") #Fixing ancilla = |0>.
            zll = "Z" + zll
        
        zll =  "-"+zll if input_state == 1 else zll
        return [*stabilizers, zll]
    
RUN_ALL = True

@pytest.mark.skipif(not RUN_ALL, reason="Testing")
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
    @pytest.mark.parametrize("stabilizer", ConcatenatedShorQubit(1).get_stabilizers())
    def test_logical_0_is_unaffected_by_stabilisers(self, stabilizer):
        #Arrange 
        sq = ConcatenatedShorQubit(1)
        qc = sq.encoder().compose(stabilizer)
        target = TestUtilities.logical_0()

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    @pytest.mark.parametrize("stabilizer", ConcatenatedShorQubit(1).get_stabilizers())
    def test_logical_1_is_unaffected_by_stabilisers(self, stabilizer):
        #Arrange 
        sq = ConcatenatedShorQubit(1)
        qc = TestUtilities.ket_1_circuit().compose(sq.encoder().compose(stabilizer))
        target = TestUtilities.logical_1()

        #Act
        final_state = qi.Statevector.from_instruction(qc)

        #Assert
        assert np.isclose(qi.state_fidelity(final_state, target), 1)

    #Assignement 3 and 4
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
        sq = ConcatenatedShorQubit(1)
        qc = sq.encoder()
        qc.add_register(AncillaRegister(1))
        qc.x(qubit)
        qc.compose(sq.syndrome_correction_circuit(correct_syndromes=False), inplace=True)

        #Act
        result = TestUtilities.aer_simulation(qc).get_counts()

        #Assert
        #There should only be a single syndrome as the simulation is noiseless
        assert len(result.keys()) == 1
        assert list(result.keys())[0] == syndrome #Bit order is reversed in qiskit

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
        sq = ConcatenatedShorQubit(1)
        qc = sq.encoder()
        qc.add_register(AncillaRegister(8))
        qc.z(qubit)
        qc.compose(sq.syndrome_correction_circuit(correct_syndromes=False), inplace=True)

        #Act
        result = TestUtilities.aer_simulation(qc).get_counts()

        #Assert
        #There should only be a single syndrome as the simulation is noiseless
        assert len(result.keys()) == 1
        assert list(result.keys())[0] == syndrome #Bit order is reversed in qiskit

    #Assignment 5
    @pytest.mark.parametrize("qubit, pauli_error, target", [(qubit, pauli_error, target) 
                                                    for qubit in range(9) 
                                                    for pauli_error in ["Y"]#, "Z", "X"] 
                                                    for target in [0, 1]])
    def test_single_pauli_errors_are_corrected(self, qubit, pauli_error, target):
        #Arrange
        sq = ConcatenatedShorQubit(1)
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

@pytest.mark.skipif(not RUN_ALL, reason="Testing")
class TestConcanetatedShorQubit():
    #@pytest.mark.skip()
    @pytest.mark.parametrize("input_state", [0, 1])
    def test_encoder_produces_logical_states(self, input_state):
        #Arrange
        #Construct target stabilizer state. 
        target = qi.StabilizerState.from_stabilizer_list(TestUtilities.logical_state_stabilizers(input_state))
        
        #Construct encoding circuit
        qc = QuantumCircuit(9**2)
        if input_state == 1:
            qc.x(0) #input |1>
        csq = ConcatenatedShorQubit(2)
        qc.compose(csq.encoder(), inplace=True)

        #Act
        encoded_logical_state = qi.StabilizerState(qc)

        #Assert
        assert encoded_logical_state.equiv(target)

    #@pytest.mark.skip()
    @pytest.mark.parametrize("input_state, stabilizer", [(input_state, stabilizer) 
                                            for input_state in [0, 1] 
                                            for stabilizer in ConcatenatedShorQubit(2).get_stabilizers()])
    def test_logical_states_are_unaffected_by_stabilizers(self, input_state, stabilizer):
        #Arrange
        #Construct target stabilizer state. 
        target = qi.StabilizerState.from_stabilizer_list(TestUtilities.logical_state_stabilizers(input_state))
        
        #Construct encoding circuit
        qc = QuantumCircuit(9**2)
        if input_state == 1:
            qc.x(0) #input |1>
        csq = ConcatenatedShorQubit(2)
        qc.compose(csq.encoder(), inplace=True)
        qc.compose(stabilizer, inplace = True)

        #Act
        encoded_logical_0 = qi.StabilizerState(qc)

        #Assert
        assert encoded_logical_0.equiv(target)

    #@pytest.mark.skip()
    @pytest.mark.parametrize("input_state, num_ys", [(input_state, num_ys) 
                                        for input_state in [0, 1] 
                                        for num_ys in [1,2,3]])
    def test_several_bit_and_phase_flips_get_corrected(self, input_state, num_ys):
        #Arrange
        target = qi.StabilizerState.from_stabilizer_list(TestUtilities.logical_state_stabilizers(input_state, include_ancilla=True))

        #Circuit
        csq = ConcatenatedShorQubit(2)
        qc = QuantumCircuit(csq.num_qubits + csq.num_ancillas)
        if input_state == 1:
            qc.x(0)
        qc.compose(csq.encoder(), qubits = range(csq.num_qubits), inplace=True)

        for i in range(num_ys):
            for n in range(9):
                qc.y(n + 9*i) #add phase flips in separate groups.
        qc.compose(csq.syndrome_correction_circuit(), inplace=True)
        qc.save_stabilizer()
        
        #Simulator
        aer = AerSimulator(method="stabilizer")
        transpiled_qc = transpile(qc, aer)

        #Act
        result = aer.run(transpiled_qc, shots = 1).result()
        final_stabilizer_state = result.data()['stabilizer']

        #Assert
        assert target.equiv(final_stabilizer_state)

pytest.mark.skipif(not RUN_ALL, reason="Testing")
class TestLogicalGates:
    @pytest.mark.parametrize("n, input_state", [(n, input_state) for n in [1, 2] for input_state in [0, 1]])
    def test_logical_X(self, n, input_state):
        #Arrange
        csq = ConcatenatedShorQubit(n)
        qc_target = QuantumCircuit(csq.num_qubits)
        if input_state == 1:
            qc_target.x(0)
        
        qc_test = qc_target.copy()

        qc_target.x(0) #Gate under test
        qc_target.compose(csq.encoder(), inplace=True)

        qc_test.compose(csq.encoder(), inplace=True)
        qc_test.compose(csq.logical_X(), inplace=True) #Logical gate

        #Act
        target_stabilizer_state = qi.StabilizerState(qc_target)
        test_stabilizer_state = qi.StabilizerState(qc_test)

        #Assert
        assert target_stabilizer_state.equiv(test_stabilizer_state)

    @pytest.mark.parametrize("n, input_state", [(n, input_state) for n in [1, 2] for input_state in [0, 1]])
    def test_logical_Z(self, n, input_state):
        #Arrange
        csq = ConcatenatedShorQubit(n)
        qc_target = QuantumCircuit(csq.num_qubits)
        if input_state == 1:
            qc_target.x(0)
        
        qc_test = qc_target.copy()

        qc_target.z(0) #Gate under test
        qc_target.compose(csq.encoder(), inplace=True)

        qc_test.compose(csq.encoder(), inplace=True)
        qc_test.compose(csq.logical_Z(), inplace=True) #Logical gate

        #Act
        target_stabilizer_state = qi.StabilizerState(qc_target)
        test_stabilizer_state = qi.StabilizerState(qc_test)

        #Assert
        assert target_stabilizer_state.equiv(test_stabilizer_state)
        
    @pytest.mark.parametrize("input_state, repetitions, use_naive, n", [(input_state, repetitions, use_naive, n) for input_state in [0, 1] for repetitions in [1,2] for use_naive in [True, False] for n in [1,2]])
    def test_logical_hadamard(self, input_state, repetitions, use_naive, n):
        #Arrange
        csq = ConcatenatedShorQubit(n)
        qc_target = QuantumCircuit(csq.num_qubits)
        if input_state == 1:
            qc_target.x(0)
        
        qc_test = qc_target.copy()

        for _ in range(repetitions):
            qc_target.h(0) #Gate under test
        qc_target.compose(csq.encoder(), inplace=True)
        qc_target.save_stabilizer()

        qc_test.compose(csq.encoder(), inplace=True)
        for _ in range(repetitions):
            qc_test.compose(csq.logical_H(use_naive = use_naive), inplace=True) #Logical gate
        qc_test.save_stabilizer()

        aer = AerSimulator(method='stabilizer')

        #Act
        target_result = aer.run(qc_target, shots = 1).result()
        test_result = aer.run(qc_test, shots = 1).result()
        target_stabilizer_state = target_result.data()['stabilizer']
        test_stabilizer_state = test_result.data()['stabilizer']

        #Assert
        assert target_stabilizer_state.equiv(test_stabilizer_state)

    @pytest.mark.parametrize("input_state, n", [(input_state, n) for input_state in [0, 1] for n in [0,1]])
    def test_logical_S(self, input_state, n):
        #Arrange
        csq = ConcatenatedShorQubit(n)
        qc_target = QuantumCircuit(csq.num_qubits)
        if input_state == 1:
            qc_target.x(0)
        qc_target.h(0)

        qc_test = qc_target.copy()

        qc_target.s(0) #Gate under test
        qc_target.compose(csq.encoder(), inplace=True)
        qc_target.save_stabilizer()

        qc_test.compose(csq.encoder(), inplace=True)
        qc_test.compose(csq.logical_S(), inplace=True) #Logical gate
        qc_test.save_stabilizer()

        aer = AerSimulator(method='stabilizer')

        #Act
        target_result = aer.run(qc_target, shots = 1).result()
        test_result = aer.run(qc_test, shots = 1).result()
        target_stabilizer_state = target_result.data()['stabilizer']
        test_stabilizer_state = test_result.data()['stabilizer']

        #Assert
        assert target_stabilizer_state.equiv(test_stabilizer_state)

@pytest.mark.skipif(RUN_ALL, reason = "Testing")
class TestShorCircuit:
    @pytest.mark.parametrize("n1, n2, keep_transversal", [(n1, n2, keep_transversal) for n1 in [0,1,2] for n2 in [0,1,2] for keep_transversal in [False, True]])
    def test_create_encoded_bell_pair(self, n1, n2, keep_transversal):
        """
        This test implicitly tests all logical gates.
        """
        #Arrange
        sc = ShorCircuit([n1,n2])
        inputs = sc.input_qubit_indices
        sc.encoder(0)
        sc.encoder(1)

        #Target is encoding of a physical Bell pair
        state_prep = QuantumCircuit(sc.num_qubits)
        state_prep.h(inputs[0])
        state_prep.cx(inputs[0],inputs[1])
        target_qc = sc.get_circuit().compose(state_prep, front=True)
        target = qi.StabilizerState(target_qc)

        #Construct circuit of logical gates to construct logical Bell pair of logical states
        sc.h(0)
        sc.cx(0,1, keep_transversal=keep_transversal)
        sc.save_stabilizer()

        aer = AerSimulator(method='stabilizer')

        #Act
        result = aer.run(sc.get_circuit().decompose(), shots = 1).result()
        final_stabilizer = result.data()['stabilizer']

        #Assert
        assert target.equiv(final_stabilizer)
    

    