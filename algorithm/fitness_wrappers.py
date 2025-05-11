import numpy as np
from algorithm.ga_operators import decode_individual
from benchmark_functions import Hypersphere
from opfunu.cec_based import cec2014

hypersphere_fun = Hypersphere(n_dimensions=10)
composition6_fun = cec2014.F282014(ndim=10)

# funkcje przystosowania (binarna reprezentacja)
def fitness_hypersphere_bin(ga, solution, idx):
    real = decode_individual(solution, bits_per_var=20, n_vars=10,
                             low=-5.12, high=5.12)
    return -hypersphere_fun(real)

def fitness_composition6_bin(ga, solution, idx):
    real = decode_individual(solution, bits_per_var=20, n_vars=10,
                             low=composition6_fun.bounds[0][0],
                             high=composition6_fun.bounds[0][1])
    return -composition6_fun.evaluate(real)


# reprezentacja rzeczywista
def fitness_hypersphere_real(ga, solution, idx):
    return -hypersphere_fun(solution)

def fitness_composition6_real(ga, solution, idx):
    return -composition6_fun.evaluate(solution)
