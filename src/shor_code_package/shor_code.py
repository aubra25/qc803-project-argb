from qiskit import QuantumCircuit, AncillaRegister, ClassicalRegister
from qiskit.quantum_info import Operator, Clifford
import numpy as np
from qiskit_aer import AerSimulator

class ShorQubit():
    def __init__(self):
        """
        Initialize the ShorCode instance.
        """
        self.num_qubits = 9 + 8

    def circuit(self):
        """
        Returns quantum circuit for initializing a state in the Shor code, measuring stabilisers, and correcting
        the syndromes.
        """
        #Initialize circuit
        n_qubits = 9
        n_ancillas = 8
        qc = QuantumCircuit(n_qubits + n_ancillas)
        
        #Compose components
        qc.compose(self.encoder(), inplace = True)
        qc.barrier()
        qc.compose(self.syndrome_correction_circuit(), clbits=range(8), inplace = True)

        return qc


    def encoder(self):
        """
        Create the encoding circuit of a single (logical) qubit to 9 physical qubits.
        The qubit whose state should be encoded in the Shor code has index 0.
        """
        #Initialize the circuit.
        qc = QuantumCircuit(9)

        #Outer code - 3 qubit bit flip code
        qc.cx(0, 3)
        qc.cx(0, 6)
        qc.barrier()

        #Inner code - 3 qubit phase flip code
        for n in [0, 3, 6]:
            qc.h(n)
            qc.cx(n, n + 1)
            qc.cx(n, n + 2)
    
        return qc
    
    def syndrome_correction_circuit(self, correct_syndromes = True):
        """
        Return a QuantumCircuit measuring the stabilizers of the Shor code.
        
        Each stabilizer has eigenvalues +1 or -1, so Quantum Phase Estimation
        can be utilized with a single ancilla per stabilizer to get the
        measurement outcome of the stabilizer non-destructively.
        """
        #Initialize circuit
        stabilizers = self.get_stabilizers()
        qc = QuantumCircuit(9)
        ancilla_register = AncillaRegister(1)
        qc.add_register(ancilla_register)

        #Register for measuring Z stabilizer
        z_register = ClassicalRegister(2, name = 'Z stabilizer measurements')
        qc.add_register(z_register)

        #Registers for measuring X stabilizers
        x_registers = []
        for i in range(3):
            x_register = ClassicalRegister(2, name = f'X stabilizer measurements {i}')
            qc.add_register(x_register)
            x_registers.append(x_register)

        #Create combined classical register
        combined_register = np.concatenate((z_register, np.array(x_registers).flatten()))

        #Add phase estimation for each stabilizer
        ancilla = ancilla_register[0]
        for classic_bit, stabilizer in zip(combined_register, stabilizers):
            cu = stabilizer.to_gate().control(1)
            qc.h(ancilla)
            qc.append(cu, [ancilla, *range(9)])
            qc.h(ancilla)
            qc.measure(ancilla, classic_bit)

            #Reset the ancilla for use in another round of error correction
            qc.reset(ancilla)

        #For testing purposes, correction of the syndromes can be disabled
        if correct_syndromes:
            qc.barrier()
            qc = self._add_syndrome_correction(qc, z_register, x_registers)

        return qc
    
    def _add_syndrome_correction(self, qc, z_register, x_registers):
        """
        Adds syndrome corretion to qc depending on the measured syndrome.
        """
        #Configure restoring action depending on syndrome
        x_corrections = [
            (0b01, 0),
            (0b11, 1),
            (0b10, 2),
        ]
        for (group, x_register) in enumerate(x_registers): #Loop through each grouping of three qubits
            for (syndrome, qubit) in x_corrections:
                with qc.if_test((x_register, syndrome)):
                    qc.x(qubit + 3*group)

        z_corrections = [
            (0b01, [0,1,2]),
            (0b11, [3,4,5]),
            (0b10, [6,7,8])
            ]
        for (syndrome, qubits) in z_corrections:
            with qc.if_test((z_register, syndrome)):
                for qubit in qubits:
                    qc.z(qubit)
    
        return qc


    def get_stabilizers(self):
        """
        Return a list of QuantumCircuits that implement the stabilizers of the Shor code.
        """
        stabilizer_circuits = []
        #The Shor Code has 8 stabilizers. First the stabilizers involving X are constructed.
        for n in [0, 3]:
            qc = QuantumCircuit(9, name=f"X_{n}X_{n+1}X_{n+2}X_{n+3}X_{n+4}X_{n+5}")
            for q in range(n, n + 6):
                qc.x(q)
            stabilizer_circuits.append(qc)
        
        #The stabilizers involving Z.
        for n in [0, 1, 3, 4, 6, 7]:
            qc = QuantumCircuit(9, name = f"Z_{n}Z_{n+1}")
            qc.z(n)
            qc.z(n+1)
            stabilizer_circuits.append(qc)
        
        return stabilizer_circuits
    
    def logical_X(self):
        """
        Returns a QuantumCircuit implementing the logical X gate for this logical qubit.
        """
        qc = QuantumCircuit(9)
        for n in range(9):
            qc.z(n)
        return qc

    def logical_Z(self):
        """
        Returns a QuantumCircuit implementing the logical Z gate for this logical qubit.
        """
        qc = QuantumCircuit(9)
        for n in range(9):
            qc.x(n)
        return qc

class ConcatenatedShorQubit:
    """
    Class for constructing concatenations of the Shor nine qubit code recursively.
    """

    def __init__(self, n):
        """
        Initializes the class. n is number of concatenations of Shor nine qubit code with itself.
        """
        self.n = n
        self.num_qubits = 9**n #physical qubits
        self.num_ancillas = 1
        self._inner_code = ConcatenatedShorQubit(n - 1) if n >= 1 else None

    def encoder(self):
        """
        Create the encoding circuit of a single (logical) qubit to 9^n physical qubits.
        The qubit whose state should be encoded in the concatenated Shor code has index 0.
        """
        #Base case
        if self.n == 0:
            return QuantumCircuit(1)

        #Initialize circuit
        qc = QuantumCircuit(self.num_qubits)
        outer_groups = np.split(np.arange(self.num_qubits), 3)
        outer_principal_qubits = [sub_indices[0] for sub_indices in outer_groups]
        
        qc.cx(outer_principal_qubits[0], outer_principal_qubits[1])
        qc.cx(outer_principal_qubits[0], outer_principal_qubits[2])
        
        for outer_group in outer_groups:
            inner_group = np.split(outer_group, 3)
            inner_principal_qubits = [group[0] for group in inner_group]

            qc.h(inner_principal_qubits[0])
            qc.cx(inner_principal_qubits[0], inner_principal_qubits[1])
            qc.cx(inner_principal_qubits[0], inner_principal_qubits[2])

            for group in inner_group:
                qc.compose(self._inner_code.encoder(), qubits = group, inplace = True)

        return qc
    
    def syndrome_correction_circuit(self, correct_syndromes = True):
        """
        Constructs the circuit for measuring syndromes and correcting the errors.
        Classical registers behave badly with recursion, so this is implemented
        more imperatively.
        """    
        #Initialize circuit and get stabilizers and recovering actions
        qc = QuantumCircuit(self.num_qubits + self.num_ancillas)
        stabilizer_circuits = self.get_stabilizers(True)[::-1]
        recovering_circuits = self.get_recovering_circuits(True)[::-1]
        classical_register_bits = ClassicalRegister(len(stabilizer_circuits))
        qc.add_register(classical_register_bits)

        ancilla = self.num_qubits #Ancilla is the last qubit.

        #Add each stabilizer to the circuit. In the Shor code the stabilizers from pairs
        #which each receive a classical register which the restoring action is conditioned on.
        for n in range(len(stabilizer_circuits)//2):
            for k in range(2):
                #Measure the stabilizer using quantum phase estimation
                cu = stabilizer_circuits[2*n + k].to_gate().control(1)
                qc.h(ancilla)
                qc.append(cu, [ancilla, *range(self.num_qubits)])
                qc.h(ancilla)
                qc.measure(ancilla, classical_register_bits[2*n + k])
                qc.reset(ancilla)

            #Correct syndromes
            if correct_syndromes:
                with qc.if_test((classical_register_bits[2*n], 1)) as _else:
                    with qc.if_test((classical_register_bits[2*n + 1], 1)) as _inner_else:
                        # 0b11
                        qc.compose(recovering_circuits[3*n + 1], inplace = True)
                    with _inner_else:
                        # 0b10
                        qc.compose(recovering_circuits[3*n + 0], inplace = True)
                with _else:
                    with qc.if_test((classical_register_bits[2*n + 1], 1)):
                        # 0b01
                        qc.compose(recovering_circuits[3*n + 2], inplace = True)    
        return qc

    def logical_X(self):
        """
        The logical X operation for this codes logical qubit representation.
        """
        #Base case
        if self.n == 0:
            qc = QuantumCircuit(1)
            qc.x(0)
            return qc
        
        #Inductive step
        groups = np.split(np.arange(self.num_qubits), 9)
        qc = QuantumCircuit(self.num_qubits)

        #The logical X gate of the Shor code is ZZZZZZZZZ
        for group in groups:
            qc.compose(self._inner_code.logical_Z(), qubits = group, inplace = True)

        return qc

    def logical_Z(self):
        """
        The logical Z operation for this codes logical qubit representation.
        """
        #Base case
        if self.n == 0:
            qc = QuantumCircuit(1)
            qc.z(0)
            return qc
        
        #Inductive step
        groups = np.split(np.arange(self.num_qubits), 9)
        qc = QuantumCircuit(self.num_qubits)

        #The logical X gate of the Shor code is XXXXXXXXX
        for group in groups:
            qc.compose(self._inner_code.logical_X(), qubits = group, inplace = True)

        return qc
    
    def logical_H(self, use_naive = False):
        """
        The logical H operation. If use_naive obtained by naively conjugating H on the input qubit with the encoder unitary.
        """
        if self.n >= 2:
            raise Exception("Not defined for n >= 2. Operation matrix would be 2**81 x 2**81")
        
        #Base case:
        if self.n == 0:
            qc = QuantumCircuit(1)
            qc.h(0)
            return qc

        #Inductive step
        if use_naive:
            groups = np.split(np.arange(self.num_qubits), 9)
            #The naive H gate of the Shor code is 1/sqrt(2)(X_0 X_1 X_2 + Z_0 Z_3 Z_6)
            #Construct operators
            inner_xl = self._inner_code.logical_X()
            inner_zl = self._inner_code.logical_Z()
            qc_X0_X1_X2 = QuantumCircuit(self.num_qubits)
            qc_Z0_Z3_Z6 = QuantumCircuit(self.num_qubits)
            for n in range(3):
                qc_X0_X1_X2.compose(inner_xl, qubits = groups[n], inplace=True)
                qc_Z0_Z3_Z6.compose(inner_zl, qubits = groups[3*n], inplace=True)

            #Combine into linear combination
            h = 1/np.sqrt(2)*(Operator(qc_X0_X1_X2) + Operator(qc_Z0_Z3_Z6))
        else:
            xl = self.logical_X()
            zl = self.logical_Z()
            h = 1/np.sqrt(2)*(Operator(xl) + Operator(zl))

        #For using in stabilizer circuits, convert operation to a Clifford gate
        #It is not trivial that the logical H gate would be a Clifford operation but it
        #turns out that it is the case!
        h_clifford = Clifford.from_operator(h)
        return h_clifford.to_circuit()
    
    def logical_S(self):
        """
        Create the logical phase gate for the code.
        """
        if self.n >= 2:
            raise Exception("Not defined for n >= 2. Operation matrix would be 2**81 x 2**81")
        #Base case:
        if self.n == 0:
            qc = QuantumCircuit(1)
            qc.s(0)
            return qc

        #Inductive step
        qc = QuantumCircuit(self.num_qubits)
        h = 1/2*((1+1j)*Operator.from_label("I"*self.num_qubits) + (1-1j)*Operator.from_label("X"*self.num_qubits))

        #For using in stabilizer circuits, convert operation to a Clifford gate
        h_clifford = Clifford.from_operator(h)
        return h_clifford.to_circuit()
        


    def get_stabilizers(self, include_inner_stabilizers = False):
        """
        Returns the stabilizers for the outer code. These are constructed using the logical X and Z operations on the inner level
        logical states.
        param include_inner_stabilizers can be set to also return inner stabilizers.
        """
        #Base case
        if self.n == 0:
            return []

        #Inductive step
        #Set up needed components for the stabilizer circuit.
        shor_code_stabilizers = ["XXXXXXIII", "IIIXXXXXX", "ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ"]
        groups = np.split(np.arange(self.num_qubits), 9) #Nine groupings of qubits.
        xl = self._inner_code.logical_X()
        zl = self._inner_code.logical_Z()

        #Construct each stabilizer circuit
        stabilizer_circuits = []
        for stabilizer in shor_code_stabilizers:
            qc = QuantumCircuit(self.num_qubits)
            for i, pauli in enumerate(stabilizer):
                match(pauli):
                    case "X":
                        qc.compose(xl, qubits = groups[i], inplace = True)
                    case "Z":
                        qc.compose(zl, qubits = groups[i], inplace = True)
                    case "I":
                        pass #Do nothing for the identity.
            stabilizer_circuits.append(qc)
        
        #Collect the stabilizers of the inner codes on each group
        inner_stabilizer_circuits = []
        if include_inner_stabilizers == True:
            inner_stabilizers = self._inner_code.get_stabilizers(True)
            for group in groups:
                for stabilizer in inner_stabilizers:
                    qc = QuantumCircuit(self.num_qubits)
                    qc.compose(stabilizer, qubits = group, inplace = True)
                    inner_stabilizer_circuits.append(qc)

        return [*stabilizer_circuits, *inner_stabilizer_circuits]
    
    def get_recovering_circuits(self, include_inner_recovering_circuits = False):
        """
        Get circuits for performing the recovery operations.
        """
        #Base case
        if self.n == 0:
            return [] #No recovering actions for a single qubit.
        
        #Inductive step
        groups = np.split(np.arange(self.num_qubits), 3)
        
        #Construct phase flip correction circuits.
        zzz_circuits = []
        for group in groups:
            qc = QuantumCircuit(self.num_qubits)
            for sub_group in np.split(group, 3):
                qc.compose(self._inner_code.logical_Z(), qubits = sub_group, inplace = True)
            zzz_circuits.append(qc)

        #Construct bit flip correction circuits
        x_circuits = []
        for group in groups:
            for sub_group in np.split(group, 3):
                qc = QuantumCircuit(self.num_qubits)
                qc.compose(self._inner_code.logical_X(), qubits = sub_group, inplace = True)
                x_circuits.append(qc)

        #Construct the recovery circuits for each of the nine "physical" qubits.
        all_inner_recovering_circuits = []
        if include_inner_recovering_circuits:
            inner_recovering_circuits = self._inner_code.get_recovering_circuits(True)
            for sub_group in np.split(np.arange(self.num_qubits), 9):
                for recovering_circuit in inner_recovering_circuits:
                    qc = QuantumCircuit(self.num_qubits)
                    qc.compose(recovering_circuit, qubits = sub_group, inplace = True)
                    all_inner_recovering_circuits.append(qc)
                    
        return [*zzz_circuits, *x_circuits, *all_inner_recovering_circuits]


class ShorCircuit:
    """
    This class handles construction of QuantumCircuits using the Shor nine qubit quantum error correction code
    """

    def __init__(self, shor_code = ShorQubit()):
        """
        Initialize the ShorCircuit. A ShorCode to use must be injected.
        """
        self.sc = shor_code
        self.aer = AerSimulator()
        self._initialize_circuit()

    def _initialize_circuit(self):
        """
        Initializises the encoder circuit
        """
        self._circuit = QuantumCircuit(self.sc.num_qubits)
        self._circuit.compose(self.sc.encoder())

    def x(self, qubit):
        """
        Add Pauli X in between encoding and syndrome measurement.
        """
        self._circuit.x(qubit)

    def y(self, qubit):
        """
        Add Pauli Y in between encoding and syndrome measurement.
        """
        self._circuit.y(qubit)

    def z(self, qubit):
        """
        Add Pauli Z in between encoding and syndrome measurement.
        """
        self._circuit.z(qubit)

    def get_circuit(self):
        """
        Returns the QuantumCircuit built using the Shor encoding.
        """
        qc = self._circuit.copy()
        qc.compose(self.sc.stabilizer_measurement_circuit(), inplace=True)
        return qc
    
    def simulate_noiseless(self, shots = 1e3):
        """
        Run a simulation of the circuit using the Aer simulator. Returns the result object of the simulation.
        """
        qc = self.get_circuit()
        result = self.aer.run(qc.decompose(), shots = shots).result()
        return result