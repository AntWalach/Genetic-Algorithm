import numpy as np
from benchmark_functions import Hyperellipsoid
from opfunu.cec_based import cec2014
from algorithm.ga_operators import decode_individual

# --- funkcje benchmark ---------------------------------------------
hfun  = Hyperellipsoid(n_dimensions=10)
f3fun = cec2014.F32014(ndim=10)

# --- BINARNA -------------------------------------------------------
def fitness_hyper_bin(ga, solution, idx):
    real = decode_individual(solution, bits_per_var=20, n_vars=10,
                             low=-32.768, high=32.768)
    return -hfun(real)

def fitness_f3_bin(solution, _):
    real = decode_individual(solution, bits_per_var=20, n_vars=10,
                             low=f3fun.bounds[0][0], high=f3fun.bounds[0][1])
    return -f3fun.evaluate(real)

# --- RZECZYWISTA ---------------------------------------------------
def fitness_hyper_real(solution, _):
    return -hfun(solution)

def fitness_f3_real(solution, _):
    return -f3fun.evaluate(solution)
