import numpy as np
from docplex.mp.model import Model
from docplex.mp.constants import ComparisonType
from helper_functions import *

def ilp_to_qubo(model):
    if not isinstance(model, Model):
        raise Exception("The input to the model should be a DOCPLEX Model.")

    num_vars = model.number_of_binary_variables
    num_constraints = model.number_of_linear_constraints

    vars = []
    constraints = []
    
    for i in range(num_vars):
        vars.append(model.get_var_by_index(i))
    
    for i in range(num_constraints):
        constraints.append(model.get_constraint_by_index(i))

    # Creating a QUBO model.
    qubo = Model(name=model.name+'_qubo')
    
    # Adding variables to the QUBO model.
    qubo.binary_var_list(vars)

    # Obtaining the objective function of the ILP model.
    objective = model.get_objective_expr()

    # Setting a large penalty term.
    penalty = len(list(model.iter_binary_vars()))
    
    # Converting the linear constraints into objective functions.
    for i in range(num_constraints):
        if constraints[i].rhs.number_of_variables() == 0 and constraints[i].sense == ComparisonType(2):
            new_constraint = ((constraints[i].lhs-1) == 0)
            objective += penalty * linear_eq_const_to_quad_obj(new_constraint)
        elif constraints[i].sense == ComparisonType(1):
            new_constraint = ((constraints[i].lhs-constraints[i].rhs) <= 0)
            bound = find_bound_for_slack(new_constraint)
            (bound, exp) = get_real_bound_in_int(bound)
            bound_len = int(np.ceil(np.log2(bound)))
            slack_expr = 0
            for j in range(bound_len):
                new_var = qubo.binary_var(name='s_'+str(i)+'_'+str(j))
                slack_expr = (2 * slack_expr) + new_var
            sub_objective = linear_eq_const_to_quad_obj(new_constraint.lhs + (slack_expr/(10**exp)) == 0)
            objective += penalty * sub_objective        

    qubo.minimize(objective)

    return qubo