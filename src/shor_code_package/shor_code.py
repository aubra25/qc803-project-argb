from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Operator, Clifford
import numpy as np
from qiskit.circuit.classical import expr
from qiskit.circuit import Gate

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
        self._cache = dict()

    def encoder(self):
        """
        Create the encoding circuit of a single (logical) qubit to 9^n physical qubits.
        The qubit whose state should be encoded in the concatenated Shor code has index 0.
        """
        #Cache lookup:
        cached_gate = self._cache.get('encoder', None)
        if cached_gate is not None:
            return cached_gate

        #Base case
        if self.n == 0:
            return QuantumCircuit(1)

        #Initialize circuit
        qc = QuantumCircuit(self.num_qubits)

        #Construct encoder in two steps. First outer grouping of 3, then 3x inner grouping per outer group.
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

            #Recursive call
            for group in inner_group:
                qc.compose(self._inner_code.encoder(), qubits = group, inplace = True)
        
        self._cache['encoder'] = qc
        return qc
    
    def syndrome_correction_circuit(self, correct_syndromes = True, num_measurements = 1, remeasure_same_qubit = False, _classical_register = None):
        """
        Constructs the circuit for measuring syndromes and correcting the errors.
        correct_syndromes can be set to control whether syndromes are simply diagnosed
        or recovery actions are performed based on the syndromes.
        Classical registers behave badly with recursion, so this is implemented
        more imperatively.
        """    
        #Cache lookup:
        cached_gate = self._cache.get('error_correction', None)
        if cached_gate is not None:
            return cached_gate
        
        #Initialize circuit and get stabilizers and recovering actions
        qc = QuantumCircuit(self.num_qubits + self.num_ancillas)
        stabilizer_circuits = self.get_stabilizers(include_inner_stabilizers=True)[::-1]
        recovering_circuits = self.get_recovering_circuits(include_inner_recovering_circuits=True)[::-1]

        classical_register_bits = ClassicalRegister(len(stabilizer_circuits)*num_measurements) if _classical_register is None else _classical_register
        qc.add_register(classical_register_bits)

        ancilla = self.num_qubits #Ancilla is the last qubit.

        #Add each stabilizer to the circuit. In the Shor code the stabilizers form pairs
        #which each receive a classical register which the restoring action is conditioned on.
        for n in range(len(stabilizer_circuits)//2): #for each pair
            for j in range(num_measurements if not remeasure_same_qubit else 1): #for each measuremnt
                for k in range(2): #for each stabilizer of the pair
                    #Measure the stabilizer using quantum phase estimation
                    cu = stabilizer_circuits[2*n + k].to_gate().control(1)
                    qc.h(ancilla)
                    qc.append(cu, [ancilla, *range(self.num_qubits)])
                    qc.h(ancilla)
                    for l in range(num_measurements if remeasure_same_qubit else 1): #for each measuremnt
                        qc.measure(ancilla, classical_register_bits[(2*n + k)*num_measurements + j + l])
                    qc.reset(ancilla)

            #Correct syndromes
            if correct_syndromes:
                #Calculate expressions for determining what recovery action to take.
                #Default is single measurement, but the expressions allow determining how to correct
                #on a majority vote between many measurements.
                expressions = []
                for index in [2*n, 2*n + 1]:
                    #Get relevant classical bits
                    register = classical_register_bits[index*num_measurements:(index+1)*num_measurements]
                    expressions.append(self._get_classical_register_majority_expression(register))

                #Set up if_tests
                with qc.if_test(expressions[0]) as _else:
                    with qc.if_test(expressions[1]) as _inner_else:
                        qc.compose(recovering_circuits[3*n + 1], inplace = True)
                    with _inner_else:
                        qc.compose(recovering_circuits[3*n + 0], inplace = True)
                with _else:
                    with qc.if_test(expressions[1]):
                        qc.compose(recovering_circuits[3*n + 2], inplace = True)    
        
        self._cache['error_correction'] = qc
        return qc
    
    def _get_classical_register_majority_expression(self, register):
        """
        Returns an expression returning True if the register has a majority of bits set to 1.
        """
        num_measurements = len(register)
        #When doing multiple measurements, majority of the measurements determine
        #the outcome. To act on this outcome, a table of binary numbers with majority
        #of 1's is generated.
        allowed_bit_strings = [i for i in range(2**num_measurements) if int.bit_count(i) > num_measurements/2]
        
        expression = expr.lift(False)
        for bit_string in allowed_bit_strings:
            #For each bit string with a majority of 1's, get equal expressions bitwise and 'and' them together.
            equals_expressions = [expr.equal(reg, bool(int(bit))) for reg, bit in zip(register, format(bit_string, f"0{num_measurements}b"))]
            clause = expr.lift(True)
            for i in range(num_measurements):
                clause = expr.bit_and(equals_expressions[i], clause)
            #'or' all the clauses together.
            expression = expr.bit_or(expression, clause)
        return expression

    def logical_X(self):
        """
        The logical X operation for this codes logical qubit representation.
        """
        #Cache lookup:
        cached_gate = self._cache.get('x', None)
        if cached_gate is not None:
            return cached_gate
        
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
        
        self._cache['x'] = qc
        return qc

    def logical_Z(self):
        """
        The logical Z operation for this codes logical qubit representation.
        """
        #Cache lookup:
        cached_gate = self._cache.get('z', None)
        if cached_gate is not None:
            return cached_gate
        
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

        self._cache['z'] = qc
        return qc
    
    def logical_H(self, use_naive = False, intermediate_error_correction = False):
        """
        Create the logical H operation. 
        If use_naive obtained by naively conjugating H on the input qubit with the encoder unitary.
        If intermediate_error_correction, add error correction placeholders in between logical operations.
        """
        #Cache lookup:
        cached_gate = self._cache.get(f'h{intermediate_error_correction}', None)
        if cached_gate is not None:
            return cached_gate

        #Base case:
        if self.n == 0:
            qc = QuantumCircuit(1)
            qc.h(0)
            return qc

        #Inductive step
        if use_naive:
            h = 1/np.sqrt(2)*(Operator.from_label(6*"I" + 3*"X") + Operator.from_label("IIZ"*3))
        else:
            h = 1/np.sqrt(2)*(Operator.from_label(9*"X") + Operator.from_label(9*"Z"))
        #For when using in stabilizer circuits (only Clifford gates), convert operation to a Clifford gate
        #It is not trivial that the logical H gate would be a Clifford operation but it
        #turns out that it is the case!
        h_clifford = Clifford.from_operator(h)
        qc = None
        if self.n == 1:
            qc = h_clifford.to_circuit()
        else:
            #If n > 1, construct the generated circuit using logical gates of inner code.
            qc = self._construct_logical_circuit(h_clifford.to_circuit(), intermediate_error_correction)
        
        self._cache[f'h{intermediate_error_correction}'] = qc
        return qc
                
    def logical_S(self, intermediate_error_correction = False):
        """
        Create the logical phase gate for the code.
        """
        #Cache lookup:
        cached_gate = self._cache.get(f's{intermediate_error_correction}', None)
        if cached_gate is not None:
            return cached_gate
        
        #Base case:
        if self.n == 0:
            qc = QuantumCircuit(1)
            qc.s(0)
            return qc

        #Inductive step
        qc = QuantumCircuit(self.num_qubits)
        s = 1/2*((1+1j)*Operator.from_label("I"*9) + (1-1j)*Operator.from_label("X"*9))

        #For using in stabilizer circuits, convert operation to a Clifford gate
        h_clifford = Clifford.from_operator(s)
        qc = None
        if self.n == 1:
            qc = h_clifford.to_circuit()
        else:
            #If n > 1, construct the generated circuit using logical gates of inner code.
            qc = self._construct_logical_circuit(h_clifford.to_circuit(), intermediate_error_correction)

        self._cache[f's{intermediate_error_correction}'] = qc
        return qc

    def _construct_logical_circuit(self, qc, error_correct):
        """
        Logical gates can be expressed as circuits of logical gates of the inner codes. This
        function takes a 9 qubit Clifford circuit and converts it to a circuit using the logical
        operations of the inner code.
        If error_correct, error correction placeholders are inserted in between logical operations.
        """
        #Compile list of instructions from qc.
        instructions = []
        for gate in qc:
            instructions.append((gate.name, [qc.find_bit(qubit).index for qubit in gate.qubits]))

        #Set up a ShorCircuit with 9 qubits encoded using the inner code (n-1).
        #Apply instructions from qc to the ShorCircuit.
        shor_circuit = ShorCircuit([self.n - 1 for _ in range(9)], include_ancilla = False, auto_encode=False)
        for instruction in instructions:
            if error_correct:
                for qubit in instruction[1]:
                    shor_circuit.error_correct(qubit)
            match instruction[0]:
                case "s":
                    shor_circuit.s(instruction[1][0])
                case "h":
                    shor_circuit.h(instruction[1][0])
                case "cx":
                    shor_circuit.cx(instruction[1][0], instruction[1][1])
                case "x":
                    shor_circuit.x(instruction[1][0])
                case "y":
                    shor_circuit.z(instruction[1][0])
                    shor_circuit.x(instruction[1][0])
                case "z":
                    shor_circuit.z(instruction[1][0])
        
        #Decompose to basis gates and error correction placeholders
        result = shor_circuit.get_circuit(transpile = False).decompose()
        return result

    def get_stabilizers(self, include_inner_stabilizers = False):
        """
        Returns the stabilizers for the outer code. These are constructed using the logical X and Z operations on the inner level
        logical states.
        include_inner_stabilizers can be set to also return stabilizers of the inner code.
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
            #The inner code stabilizers apply to each of the 9 groups individually.
            for group in groups:
                for stabilizer in inner_stabilizers:
                    qc = QuantumCircuit(self.num_qubits)
                    qc.compose(stabilizer, qubits = group, inplace = True)
                    inner_stabilizer_circuits.append(qc)

        return [*stabilizer_circuits, *inner_stabilizer_circuits]
    
    def get_recovering_circuits(self, include_inner_recovering_circuits = False):
        """
        Get circuits for performing the recovery operations.
        include_inner_recovering_circuits can be set to also include
        recovery operations of the inner code.
        """
        #Base case
        if self.n == 0:
            return [] #No recovering actions for a single qubit.
        
        #Inductive step
        #Construct phase flip correction circuits.
        outer_groups = np.split(np.arange(self.num_qubits), 3)
        zzz_circuits = []
        for outer_group in outer_groups:
            qc = QuantumCircuit(self.num_qubits)
            for inner_group in np.split(outer_group, 3):
                qc.compose(self._inner_code.logical_Z(), qubits = inner_group, inplace = True)
            zzz_circuits.append(qc)

        #Construct bit flip correction circuits
        x_circuits = []
        for outer_group in outer_groups:
            for inner_group in np.split(outer_group, 3):
                qc = QuantumCircuit(self.num_qubits)
                qc.compose(self._inner_code.logical_X(), qubits = inner_group, inplace = True)
                x_circuits.append(qc)

        #Construct the recovery circuits for each of the nine inner qubits.
        all_inner_recovering_circuits = []
        if include_inner_recovering_circuits:
            inner_recovering_circuits = self._inner_code.get_recovering_circuits(True)
            for group in np.split(np.arange(self.num_qubits), 9):
                for recovering_circuit in inner_recovering_circuits:
                    qc = QuantumCircuit(self.num_qubits)
                    qc.compose(recovering_circuit, qubits = group, inplace = True)
                    all_inner_recovering_circuits.append(qc)
                    
        return [*zzz_circuits, *x_circuits, *all_inner_recovering_circuits]

class ShorCircuit:
    """
    This class handles construction of QuantumCircuits using the concatenated Shor nine qubit quantum error correction code.
    """

    def __init__(self, qubit_code_depths, include_ancilla = True, auto_encode = True, num_measurements_in_error_correction = 1):
        """
        Initialize the ShorCircuit. Each qubit has a ConcatenatedShorQubit encoding used for performing gates.
        qubit_code_depths: An array of integers defining each qubits encoding level (n)
        include_ancilla: Whether to include measurement ancilla by default.
        auto_encode: whether to automatically add encoders to the circuit when initialising.
        num_measurements_in_error_correction: Number of round of measurement to do majority vote between.
        """
        #Only generate as many inner codes as there are different encoding depths.
        self._inner_codes = dict([(n, ConcatenatedShorQubit(n)) for n in set(qubit_code_depths)])
        self.codes = [self._inner_codes[n] for n in qubit_code_depths]
        self.num_logical_qubits = len(self.codes)
        self.num_qubits = sum([c.num_qubits for c in self.codes])
        self.num_ancillas = 1 if include_ancilla else 0
        self.max_n = max([code.n for code in self.codes])
        self.num_classical_bits = sum([8*(9**k) for k in range(self.max_n)])
        self.num_measurements_in_error_correction = num_measurements_in_error_correction

        #Save indices of physical qubits for each logical qubit
        acc = 0
        self.qubit_indices = []
        for code in self.codes:
            self.qubit_indices.append(range(acc, acc + code.num_qubits))
            acc += code.num_qubits
        self.input_qubit_indices = [q[0] for q in self.qubit_indices]

        self.ancilla = self.num_qubits #Ancilla is the last qubit of the circuit.

        #Setup circuit and cache
        self._circuit = QuantumCircuit(self.num_qubits + self.num_ancillas)
        if auto_encode:
            for q in range(self.num_logical_qubits):
                self.encoder(q)   
        self._cache = dict()

    def encoder(self, qubit):
        """
        Add the encoder from the input qubit to its logical representation.
        """
        encoder = self.codes[qubit].encoder().to_gate()
        encoder.name = f"Encoder{qubit}"
        self._circuit.compose(encoder, qubits = self.qubit_indices[qubit], inplace=True)

    def decoder(self, qubit):
        """
        Add the decoder from the logical representation of specified qubit to its physical qubit counterpart.
        """
        decoder = self.codes[qubit].encoder().inverse().to_gate()
        decoder.name = f"Decoder{qubit}"
        self._circuit.compose(decoder, qubits = self.qubit_indices[qubit], inplace=True)
    
    def measure(self, qubit, classical_register):
        """
        Measure the specified logical qubit to the provided classical register.
        """
        #Decode the logical qubit.
        self.decoder(qubit)
        self._circuit.measure(self.input_qubit_indices[qubit], classical_register)

    def x(self, qubit):
        """
        Add logical X to logical qubit.
        """
        xl = self.codes[qubit].logical_X().to_gate()
        xl.name = f"X_logical_{qubit}"
        qubit_indices = self.qubit_indices[qubit]
        self._circuit.compose(xl, qubits = qubit_indices, inplace=True)
    
    def z(self, qubit):
        """
        Add logical Z to logical qubit.
        """
        zl = self.codes[qubit].logical_Z().to_gate()
        zl.name = f"Z_logical_{qubit}"
        qubit_indices = self.qubit_indices[qubit]
        self._circuit.compose(zl, qubits = qubit_indices, inplace=True)
    
    def h(self, qubit, intermediate_error_correction = True):
        """
        Add logical H to logical qubit.
        """
        hl = self.codes[qubit].logical_H(intermediate_error_correction=intermediate_error_correction).to_gate()
        hl.name = f"H_logical_{qubit}"
        qubit_indices = self.qubit_indices[qubit]
        self._circuit.compose(hl, qubits = qubit_indices, inplace=True)

    def s(self, qubit, intermediate_error_correction = True):
        """
        Add logical S to logical qubit.
        """
        sl = self.codes[qubit].logical_S(intermediate_error_correction=intermediate_error_correction).to_gate()
        sl.name = f"S_logical_{qubit}"
        qubit_indices = self.qubit_indices[qubit]
        self._circuit.compose(sl, qubits = qubit_indices, inplace=True)

    def sdg(self, qubit):
        """
        Add logical S dagger to logical qubit.
        """
        sl = self.codes[qubit].logical_S()
        sdgl = sl.compose(self.codes[qubit].logical_Z())
        sdgl.name = f"S_dag_logical_{qubit}"
        qubit_indices = self.qubit_indices[qubit]
        self._circuit.compose(sdgl, qubits = qubit_indices, inplace=True)

    def cx(self, control, target, keep_transversal = False):
        """
        Add a transversal logical CNOT between a control and target logical qubit.
        """
        cxl = self._logical_cx(control, target, keep_transversal)
        self._circuit.compose(cxl, inplace=True)

    def error_correct(self, qubit):
        """
        Perform error correction on the qubit.
        """
        code = self.codes[qubit]
        qubit_indices = self.qubit_indices[qubit]
        self._circuit.compose(ErrorCorrect(code.num_qubits, f"Error_correct_{code.n}"), qubits=qubit_indices, inplace=True)

    def _logical_cx(self, control, target, keep_transversal):
        """
        Construct a logical CNOT gate between two logical qubits.
        keep_transversal can be used to keep a transversal implementation when
        targets n is smaller than controls n.
        """
        #Cache lookup:
        cached_gate = self._cache.get(f'cx_{control}{target}', None)
        if cached_gate is not None:
            return cached_gate

        #Get relevant resources
        control_code = self.codes[control]
        target_code = self.codes[target]
        control_indices = self.qubit_indices[control]
        target_indices = self.qubit_indices[target]

        #Construct gates for acting on individual physical qubits depending on sizes of codes
        #being acted between.
        cx_logical = QuantumCircuit(2)
        if control_code.n % 2 == 0 and target_code.n % 2 == 0:
            cx_logical.h(1)
            cx_logical.cz(1,0)
            cx_logical.h(1)
        if control_code.n % 2 == 1 and target_code.n % 2 == 0:
            cx_logical.h(1)
            cx_logical.cx(1,0)
            cx_logical.h(1)
        if control_code.n % 2 == 0 and target_code.n % 2 == 1:
            cx_logical.cz(1,0)
        if control_code.n % 2 == 1 and target_code.n % 2 == 1:
            cx_logical.cx(1,0)

        #Construct full logical CNOT
        cx = QuantumCircuit(self.num_qubits + self.num_ancillas)
        #The gate will look differently depending on the number of physical qubits for each logical qubit.
        if control_code.num_qubits == target_code.num_qubits:
            #Transverse implementation
            for n in range(len(control_indices)):
                qubits = [control_indices[n], target_indices[n]]
                cx.compose(cx_logical, qubits = qubits, inplace=True)
        
        if control_code.num_qubits < target_code.num_qubits:
            #Each control qubit controls a group of the target.
            num_target_qubits = target_code.num_qubits // control_code.num_qubits
            for n in range(control_code.num_qubits):
                for k in range(num_target_qubits):
                    cx.compose(cx_logical, qubits = [control_indices[n], target_indices[num_target_qubits*n + k]], inplace=True)

        if control_code.num_qubits > target_code.num_qubits:
            #If an entangling gate is added per control qubit it will only achieve propagating errors from
            #all control qubits without any benefit over simply adding an entangling gate to a single
            #of the qubits. Both options are implemented to allow experimentation with this.
            num_control_qubits = control_code.num_qubits // target_code.num_qubits
            if keep_transversal:
                for k in range(target_code.num_qubits):
                    for n in range(num_control_qubits):
                        cx.compose(cx_logical, qubits = [control_indices[num_control_qubits*k + n], target_indices[k]], inplace=True)
            else:
                for k in range(0, target_code.num_qubits, max(target_code.num_qubits//3, 1)):
                    for n in range(control_code.num_qubits//3):
                        cx.compose(cx_logical, qubits = [control_indices[n], target_indices[k]], inplace=True)

        cx_gate = cx.to_gate()
        cx_gate.name = f"CX_logical_({control})->({target})"

        self._cache[f'cx_{control}{target}'] = cx_gate
        return cx_gate

    def get_circuit(self, transpile = True, classical_register = None):
        """
        Returns the QuantumCircuit built using the Shor encoding. If transpile is false, error correcting
        circuits are not inserted in the placeholder locations. Classical register can be optionally provided
        (mostly useful when num_measurements > 1 since then the classical register automatically created
        is not large enough).
        """
        return self._transpile(classical_register=classical_register) if transpile else self._circuit.copy()

    def _transpile(self, classical_register = None):
        """
        Replaces placeholder Error Correct instructions with actual error correction circuits.
        If a classical register is passed, this will be used for syndrome measurements throughout.
        """
        original_circuit = self._circuit.copy()
        original_circuit = original_circuit.decompose()
        classical_register = ClassicalRegister(self.num_classical_bits) if classical_register is None else classical_register
        
        #Find error correction placeholders in the CircuitInstructions.
        error_corrections = []
        for index, circuit_instruction in enumerate(original_circuit._data):
            if circuit_instruction.name == "ErrorCorrect":
                qubits = circuit_instruction.qubits
                n = int(circuit_instruction.label[-1]) #Last character is depth
                error_corrections.append((index, n, qubits))
        
        #Reconstruct the circuit with error correction inserted.
        previous_index = -1
        final_index = len(original_circuit._data)
        new_circuit = self._circuit.copy()
        new_circuit.clear()
        for error_correction in [*error_corrections, (final_index,None,None)]:
            index, n, qubits = error_correction
            for instruction in original_circuit._data[previous_index + 1 : index]: #Skip the ErrorCorrect instruction
                new_circuit._data.append(instruction)
            previous_index = index
            if index == final_index:
                break
            new_circuit.compose(ConcatenatedShorQubit(n).syndrome_correction_circuit(_classical_register=classical_register, num_measurements=self.num_measurements_in_error_correction), 
                                  qubits = [*qubits, self.ancilla], 
                                  inplace=True)

        return new_circuit

    def save_stabilizer(self, **kwargs):
        """
        Saves the stabilizer state for retrieval after simulation.
        """
        self._circuit.save_stabilizer(**kwargs)

    def set_stabilizer(self, stabilizer_state):
        """
        Set the the stabilizer state of the circuit to a state described by the output of the provided
        QuantumCircuit.
        """
        self._circuit.set_stabilizer(stabilizer_state)

    def add_register(self, register, **kwargs):
        """
        Add a register to the underlying QuantumCircuit.
        """
        self._circuit.add_register(register, **kwargs)
    
    def barrier(self):
        "Apply a barrier in the circuit."
        self._circuit.barrier()

    def draw(self, **kwargs):
        """
        Draw the ShorCircuit.
        """
        return self._circuit.draw(**kwargs)
    
    def if_test(self, **kwargs):
        """
        Delegates to if_test method of the QuantumCircuit class.
        """
        return self._circuit.if_test(**kwargs)


class ErrorCorrect(Gate):
    """
    Instruction acting as a placeholder for ErrorCorrection.
    """
    def __init__(self, num_qubits, label):
        self.name = "ErrorCorrect"
        super().__init__(self.name, num_qubits, [], label = label)

    def inverse(self):
        return ErrorCorrect(self.name, self.num_qubits)