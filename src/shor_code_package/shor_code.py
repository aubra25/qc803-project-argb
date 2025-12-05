from qiskit import QuantumCircuit
import numpy as np

class ShorCode():
    def __init__(self):
        """
        Initialize the ShorCode instance.
        """
        self._circuit = QuantumCircuit(9)
        print("Hi there!")

    def encoder(self):
        """
        Create the encoding circuit of a single (logical) qubit to 9 physical qubits.
        The qubit whose state should be encoded in the Shor code has index 0.
        """
        #Initialize the circuit.
        n_qubits = 9
        qc = QuantumCircuit(n_qubits)

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
        


