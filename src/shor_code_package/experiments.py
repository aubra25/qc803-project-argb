from qiskit_aer import AerSimulator
from qiskit import generate_preset_pass_manager

class NoiseModelExperiment:
    def __init__(self, qc, noise_model_generator, noise_model_generator_inputs, shots = 1000, silent = True):
        self.noise_model_generator = noise_model_generator
        self.inputs = noise_model_generator_inputs
        self.output = []
        self.aer = AerSimulator(method = "stabilizer", basis_gates = ['cx','h','s','x','z'])
        self.shots = shots

        #Transpile the circuit for the simulator
        pass_manager = generate_preset_pass_manager(1, self.aer) #Optimization level = 1 or else it won't work.
        self.qc = pass_manager.run(qc)
        self.silent = silent

    def run(self):
        for index, input in enumerate(self.inputs):
            noise_model = self.noise_model_generator(input)
            self.aer.set_option("noise_model", noise_model)
            run = self.aer.run(self.qc, shots = self.shots)
            self.output.append(run.result())
            if not self.silent:
                print(f"Simulation {index} of {len(self.inputs)} done.", end = "\r")
        return self.output
    