import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_algorithms.optimizers import SPSA
from ansatz import *


def get_var_qubit_map(model):
    ''' The function return the variable-qubit mapping
    as a dictionary.
    '''
    
    var_list = list(model.iter_variables())
    var_qubit_map = {}
    for idx, var in zip(range(len(var_list)), var_list):
        var_qubit_map[var.name] = idx

    return var_qubit_map


def set_string_val(s, i):
    string = list(s)
    string[i] = '1'

    return "".join(string)


def bool_to_pauli(s):
    ''' Given a Boolean string *s*, this function converts
    it into a string that contains 'I' in place of '0' 
    and 'Z' in place of '1' in *s*.
    '''
    
    new_s = ''
    for i in s:
        if i == '0':
            new_s += 'I'
        elif i == '1':
            new_s += 'Z'
        else:
            print('Incorrect bool string format.')
    return new_s


def qubo_to_hamiltonian(model):
    ''' The function returns the Hamiltonian corresponding to 
    the QUBO. The input to the function is a DOcplex model.
    '''
    
    obj = model.get_objective_expr()

    op_list = []
    var_list = list(model.iter_variables())
    var_qubit_map = get_var_qubit_map(model)
    
    iden = '0'*len(var_list)
    iden_string = bool_to_pauli('0'*len(var_list))[::-1]
    op_list.append(obj.constant*(SparsePauliOp([Pauli(iden_string)])))
    
    # Compute the Hamiltonian corresponding to each of the linear 
    # terms in the QUBO.
    for var, coef in obj.iter_terms():
        pauli_string = bool_to_pauli(set_string_val(str(iden), var_qubit_map[var.name]))[::-1]
        op = (coef/2)*SparsePauliOp.sum([SparsePauliOp([Pauli(iden_string)]), -1*SparsePauliOp([Pauli(pauli_string)])])
        op_list.append(op)
    
    # Compute the Hamiltonian corresponding to each of the quadratic 
    # terms in the QUBO.
    for var1, var2, coef in obj.iter_quad_triplets():
        pauli_string_v1 = bool_to_pauli(set_string_val(str(iden), var_qubit_map[var1.name]))[::-1]
        op1 = (1/2)*SparsePauliOp.sum([SparsePauliOp([Pauli(iden_string)]), -1*SparsePauliOp([Pauli(pauli_string_v1)])])
    
        pauli_string_v2 = bool_to_pauli(set_string_val(str(iden), var_qubit_map[var2.name]))[::-1]
        op2 = (1/2)*SparsePauliOp.sum([SparsePauliOp([Pauli(iden_string)]), -1*SparsePauliOp([Pauli(pauli_string_v2)])])
    
        op = coef*(op1 @ op2)
        op_list.append(op)
    
    # Compute the complete Hamiltonian corresponding to the QUBO.
    hamiltonian = SparsePauliOp.simplify(SparsePauliOp.sum(op_list))

    return hamiltonian


def solve_qubo_using_vqe(ansatz, model): 
    ''' Given an ansatz and a QUBO model, the function solves the QUBO
    usnig variational circuits and returns the optimal objective value,
    the optimal parameters and the optimal mapping. We use the SPSA
    optimizer for classical optimization.
    '''
    var_qubit_map = get_var_qubit_map(model)
    
    hamiltonian = qubo_to_hamiltonian(model)

    estimator = StatevectorEstimator()

    # Define a loss function using ansatz and Hamiltonian.
    def loss_function(params):
        job = estimator.run([(ansatz, [hamiltonian], params)])
        return job.result()[0].data.evs[0]

    final_params = None
    min_obj_val = np.inf
    for iter in range(1):
        initial_params = np.random.random(ansatz.num_parameters)
        # We use the SPSA optimizer.
        optimizer = SPSA(maxiter=6000)
        result = optimizer.minimize(loss_function, x0=initial_params)
    
        # Updating the current optimal if the new value is less
        # than the current optimal.
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

    return min_obj_val, final_params, final_mapping_dict