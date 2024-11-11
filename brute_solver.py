import numpy as np
from docplex.mp.quad import QuadExpr


def compute_solution(obj, soln):
    ''' Given an objective function *obj* and a solution mapping dict *soln*
    this function computes the value of *obj* using the assignment *soln*
    '''
    
    if len(set(obj.iter_variables())) != len(soln):
        print(obj.number_of_variables())
        print("Number of variables in objective function and solution not matching!")
        return
    value = obj.constant
    for term in obj.iter_terms():
        value += term[1]*soln[term[0].name]
    if isinstance(obj, QuadExpr):
        for term in obj.iter_quad_triplets():
            value += term[2]*(soln[term[0].name]*soln[term[1].name])

    return value


def brute_force_solver(model):
    ''' The brute force solver function. This function iterates over all
    possible assignments of the variables in *model* and outputs the 
    optimal value and the optimal assignment of the variables.
    '''
    obj = model.get_objective_expr()
    num_vars = model.number_of_variables
    model_vars = list(model.iter_binary_vars())
    solution_dict = {}
    optimal_value = np.inf
    optimal_solution = None

    # Iterating over all possible 2**n assignments.
    for i in range(2**num_vars):
        soln = bin(i)[2:].zfill(num_vars)
        
        # Constructing the solution assignment dictionary.
        for (j, var) in zip(range(num_vars), model_vars):
            solution_dict[var.name] = int(soln[j])
            
        # Using compute_solution function to obtain the value
        # of the objective given the assignment.
        obj_value = compute_solution(obj, solution_dict)
        
        # If the new value is better than the current optimal
        # then update the current optimal and the current solution.
        if obj_value < optimal_value:
            optimal_value = obj_value
            optimal_solution = solution_dict.copy()

    return optimal_value, optimal_solution