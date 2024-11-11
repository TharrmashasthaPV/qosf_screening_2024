import numpy as np
from docplex.mp.model import Model
from docplex.mp.constants import ComparisonType


def quad_term(var, vars_list):
    obj_term = var
    for i in var_list:
        if i != var:
            obj_term = term * (1-i)
    return obj_term


def linear_eq_const_to_quad_obj(constraint):
    ''' This function converts a linear equality expression into
    and equivalent quadratic constaint to add to the objective function
    of QUBO.
    '''
    return simplify_quadratic_expr((constraint.lhs)**2)


def find_bound_for_slack(constraint):
    ''' This function computes the maximum value that the slack variable
    has to take for the inequality constraint to be converted to an
    equality constraint
    '''
    bound = 0
    for var in constraint.iter_variables():
        if constraint.lhs.get_coef(var) < 0:
            bound += abs(constraint.lhs.get_coef(var))
    return bound


def get_real_bound_in_int(bound):
    ''' A function to convert a real number into the format
    z*exp(y) where z and y are integers.
    '''
    exp = 0
    while bound-int(bound) != 0 and exp<20:
        bound = 10*bound
        exp += 1

    if bound-int(bound) != 0:
        print("Please provide variables with smaller number of decimals.")
        return
    
    return (int(bound), exp)


def simplify_quadratic_expr(expr):
    ''' A function to simplify a quadratic expression by replacing the
    x_i*x_i terms with x_i.
    '''
    linear_part = expr.linear_part
    quad_part = expr - linear_part
    new_quad = 0
    for term in quad_part.iter_quad_triplets():
        if term[0].equals(term[1]):
            new_quad += term[2]*term[0]
        else:
            new_quad += term[2]*(term[0]*term[1])
    return new_quad + linear_part