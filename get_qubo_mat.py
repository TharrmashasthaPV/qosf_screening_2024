from collections import defaultdict

def get_q_matrix_dict(model):
    obj = model.get_objective_expr()

    offset = obj.constant
    q_dict = defaultdict(float)
    for (var1, var2, coef) in obj.iter_quad_triplets():
        q_dict[(var1.name, var2.name)] = coef
    for (var, coef) in obj.iter_terms():
        q_dict[(var.name, var.name)] = coef

    return q_dict, offset