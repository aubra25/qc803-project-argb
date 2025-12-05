from qiskit import QuantumCircuit, AncillaRegister, QuantumRegister, ClassicalRegister
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
        z_register = ClassicalRegister(2, name = 'Z stabilizer measurements')
        qc.add_register(z_register)

        x_registers = []
        for i in range(3):
            x_register = ClassicalRegister(2, name = f'X stabilizer measurements {i}')
            qc.add_register(x_register)
            x_registers.append(x_register)

        #Add phase estimation for each stabilizer
        ancilla = ancilla_register[0]
        for classic_bit, stabilizer in zip([*z_register, *[c for cs in x_registers for c in cs]], stabilizers):
            cu = stabilizer.to_gate().control(1)
            qc.h(ancilla)
            qc.append(cu, [ancilla, *range(9)])
            qc.h(ancilla)
            qc.measure(ancilla, classic_bit)

            #Reset the ancilla for use in another round of error correction
            qc.reset(ancilla)

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
        for (group, x_register) in enumerate(x_registers):
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


    

    
    




