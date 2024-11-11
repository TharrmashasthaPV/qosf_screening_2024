import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from vqe_solver import qubo_to_hamiltonian
from qiskit_algorithms.optimizers import SPSA


def get_var_qubit_map(model):
    ''' The function return the variable-qubit mapping
    as a dictionary.
    '''
    
    var_list = list(model.iter_variables())
    var_qubit_map = {}
    for idx, var in zip(range(len(var_list)), var_list):
        var_qubit_map[var.name] = idx

    return var_qubit_map


def create_start_state_circ(num_qubits):
    ''' The function returns a circuit that prepares
    the state |+>^{\otimes n} .
    '''
    
    circ = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circ.h(i)

    return circ


def create_mixer_circuits(num_qubits, params):
    ''' The function returns a list of circuits corresponding
    to the mixer Hamiltonian.    
    '''
    
    m_circ_list = []
    for param in params:
        mixer_circ = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            mixer_circ.rx(2*param, i)
        m_circ_list.append(mixer_circ)
    return m_circ_list


def create_problem_circuit(num_qubits, model, params):
    ''' The function reuturns a list of circuits corresponding
    to the problem Hamiltonian.
    '''
    
    qubo_obj = model.get_objective_expr()
    var_qubit_map = get_var_qubit_map(model)
    problem_circs_list = []
    for param in params:
        problem_circ = QuantumCircuit(num_qubits)
        for var, coef in qubo_obj.iter_terms():
            qubit = var_qubit_map[var.name]
            problem_circ.rz(2*param*coef, qubit)

        for var1, var2, coef in qubo_obj.iter_quad_triplets():
            qubit1 = var_qubit_map[var1.name]
            qubit2 = var_qubit_map[var2.name]
            problem_circ.cx(qubit1, qubit2)
            problem_circ.rz(2*param*coef, qubit2)
            problem_circ.cx(qubit1, qubit2)

        problem_circs_list.append(problem_circ)

    return problem_circs_list
    

def create_qaoa_circuit(model, p):
    ''' The function creates the QAOA ansatz circuit corresponding
    to the QUBO in *model* with *p* intervals.
    '''

    obj = model.get_objective_expr()
    num_qubits = model.number_of_variables
    num_params = p
    beta = ParameterVector('b',num_params)
    gamma = ParameterVector('g',num_params)

    problem_circuits = create_problem_circuit(num_qubits, model, gamma)
    mixer_circuits = create_mixer_circuits(num_qubits, beta)

    ansatz = QuantumCircuit(num_qubits)
    ansatz.append(create_start_state_circ(num_qubits), list(range(num_qubits)))
    for i in range(p):
        ansatz.append(problem_circuits[i], list(range(num_qubits)))
        ansatz.append(mixer_circuits[i], list(range(num_qubits)))

    return ansatz

    
def solve_qaoa(model, p):
    ''' The function solves the QUBO in *model* using QAOA.
    '''

    var_qubit_map = get_var_qubit_map(model)
    
    ansatz = create_qaoa_circuit(model, p)
    hamiltonian = qubo_to_hamiltonian(model)

    estimator = StatevectorEstimator()

    # Defining the loss function.
    def loss_function(params):
        job = estimator.run([(ansatz, [hamiltonian], params)])
        return job.result()[0].data.evs[0]

    final_params = None
    min_obj_val = np.inf
    for iter in range(1):
        initial_params = np.random.random(ansatz.num_parameters)
        # We use the SPSA optimizer for optimization.
        optimizer = SPSA(maxiter=3000)
        result = optimizer.minimize(loss_function, x0=initial_params)
    
        # Updating the current optimal appropriately.
        if result.fun < min_obj_val:
            min_obj_val = result.fun
            final_params = result.x

    # Obtaining the eigenstate corresponding to the parameters returned 
    # by the optimizer.
    sampler = StatevectorSampler()
    
    circ = ansatz.copy()
    circ.measure_all()
    
    sample_job = sampler.run([(circ, final_params)], shots=1000)
    sampler_result = sample_job.result()[0]
    counts = sampler_result.data['meas'].get_counts()

    max_count = 0
    max_count_state = ''
    for count in counts:
        if counts[count] > max_count:
            max_count_state = count

    # Computing the optimal variable assignment based on the obtained eigenstate.
    final_mapping_dict = {}
    for var in var_qubit_map:
        qubit = var_qubit_map[var]
        if max_count_state[len(max_count_state)-qubit-1] == '1':
            final_mapping_dict[var] = 1
        else:
            final_mapping_dict[var] = 0

    return min_obj_val, max_count_state, final_mapping_dict