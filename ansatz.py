from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import TwoLocal, NLocal


def create_sample_ansatz1(num_qubits, reps=2):
    ''' Creating a sample ansatz using the RX and CX gates.
    The ansatz contains *reps* layers of RX gates followed by a linear
    later of CX gates. In addition, there is one final later of 
    RX gates applied.
    '''
    
    ansatz = QuantumCircuit(num_qubits)

    params = ParameterVector('t', num_qubits*(reps+1))
    for j in range(reps):
        for idx in range(num_qubits):
            ansatz.rx(params[(j*num_qubits)+idx], idx)
        for idx in range(num_qubits-1):
            ansatz.cz(idx, idx+1)
    for idx in range(num_qubits):
            ansatz.rx(params[(reps*num_qubits)+idx], idx)

    return ansatz


def create_sample_ansatz2(num_qubits, reps=2):
    ''' Creating a sample ansatz using the RY and CZ gates.
    The ansatz circuit contains *reps* layers of RY gates followed by a
    linear layer of CZ gates. In addition, there is one final later of 
    RX gates applied.
    '''
    
    ansatz = QuantumCircuit(num_qubits)

    params = ParameterVector('t', num_qubits*(reps+1))
    for j in range(reps):
        for idx in range(num_qubits):
            ansatz.ry(params[(j*num_qubits)+idx], idx)
        for idx in range(num_qubits-1):
            ansatz.cz(idx, idx+1)
    for idx in range(num_qubits):
            ansatz.ry(params[(reps*num_qubits)+idx], idx)

    return ansatz


def create_twolocal_rx_rz_cx_linear_ansatz(num_qubits, reps=2):
    ''' Using the TwoLocal function available in Qiskit to construct
    an ansatz with RX and RZ gates in the rotation layers and CX gates
    in entangling layers. The qubits are entangled linearly.
    '''
    
    ansatz = TwoLocal(
        num_qubits = num_qubits,
        rotation_blocks = ['rx', 'rz'],
        entanglement_blocks = ['cx'],
        entanglement = 'linear',
        reps = reps
    )

    return ansatz

def create_twolocal_rx_ry_cz_circular_ansatz(num_qubits, reps=2):
    ''' Using the TwoLocal function available in Qiskit to construct
    an ansatz with RX and RY gates in the rotation layers and CZ gates
    in entangling layers. The qubits are entangled circularly.
    '''
    
    ansatz = TwoLocal(
        num_qubits = num_qubits,
        rotation_blocks = ['rx', 'ry'],
        entanglement_blocks = ['cz'],
        entanglement = 'linear',
        reps = reps
    )

    return ansatz

def create_twolocal_ry_rz_cx_full_ansatz(num_qubits, reps=2):
    ''' Using the TwoLocal function available in Qiskit to construct
    an ansatz with RY and RZ gates in the rotation layers and CX gates
    in entangling layers. The qubits are entangled fully.
    '''
    
    ansatz = TwoLocal(
        num_qubits = num_qubits,
        rotation_blocks = ['ry', 'rz'],
        entanglement_blocks = ['cx'],
        entanglement = 'full',
        reps = reps
    )

    return ansatz

def create_problem_specific_ansatz_rx(num_qubits, reps=1):
    ''' A function to create a problem-specific ansatz. Since the solution
    to the bin packing problem is a bit-string, this ansatz contains 
    only one layer of parametric RX gates.
    '''
    
    ansatz = QuantumCircuit(num_qubits)
    params = ParameterVector('t', num_qubits)
    for i in range(num_qubits):
        ansatz.rx(params[i], i)

    return ansatz  