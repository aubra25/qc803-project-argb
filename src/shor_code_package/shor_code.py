from qiskit import QuantumCircuit
import numpy as np

class ShorCode():
    def __init__(self):
        """
        Initialize the ShorCode instance.
        """
        self._circuit = QuantumCircuit(9)

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
    
    def get_stabilizers(self):
        """
        Return a list of QuantumCircuits that implement the stabilizers of the Shor code.
        """
        stabilizer_circuits = []
        #The Shor Code has 8 stabilizers. First the stabilizers involving X are constructed.
        for n in [0, 3]:
            qc = QuantumCircuit(9)
            for q in range(n, n + 6):
                qc.x(q)
            stabilizer_circuits.append(qc)
        
        #The stabilizers involving Z.
        for n in [0, 1, 3, 4, 6, 7]:
            qc = QuantumCircuit(9)
            qc.z(n)
            qc.z(n+1)
            stabilizer_circuits.append(qc)
        
        return stabilizer_circuits
        
        


