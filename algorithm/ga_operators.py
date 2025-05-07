"""
Własne operatory do użycia w PyGAD.
Wersja 'binary' działa na int (0/1); 'real' działa na float.
"""
import numpy as np, random

# ---------- POMOCNICZE ----------------------------------------------------
def decode_bits(bits, bits_per_var, low, high):
    """01… → float w [low, high]"""
    int_val = int("".join(map(str, bits)), 2)
    max_int = 2**bits_per_var - 1
    return low + (high - low) * int_val / max_int


def decode_individual(bit_arr, bits_per_var=20, n_vars=10, low=-32.768, high=32.768):
    """Dekoduje cały chromosom binarny do listy floatów."""
    return np.array([
        decode_bits(bit_arr[i*bits_per_var:(i+1)*bits_per_var],
                    bits_per_var, low, high)
        for i in range(n_vars)
    ])

# ---------- KRZYŻOWANIA (binary) ------------------------------------------
def one_point_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]].copy()
        p2 = parents[(k+1) % parents.shape[0]].copy()
        cp = np.random.randint(1, parents.shape[1])
        offspring[k, :cp] = p1[:cp]
        offspring[k, cp:] = p2[cp:]
    return offspring


def two_point_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]].copy()
        p2 = parents[(k+1) % parents.shape[0]].copy()
        pt1, pt2 = sorted(random.sample(range(1, parents.shape[1]), 2))
        offspring[k, :pt1]  = p1[:pt1]
        offspring[k, pt1:pt2] = p2[pt1:pt2]
        offspring[k, pt2:]  = p1[pt2:]
    return offspring


def uniform_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size, dtype=parents.dtype)
    mask = np.random.rand(*offspring_size) < 0.5
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k+1) % parents.shape[0]]
        offspring[k] = np.where(mask[k], p1, p2)
    return offspring

# ---------- MUTACJE (binary) ----------------------------------------------
def mutation_swap(offspring, ga_instance):
    for chrom in offspring:
        i, j = random.sample(range(chrom.size), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return offspring


def mutation_random_bit(offspring, ga_instance):
    for chrom in offspring:
        idx = np.random.randint(chrom.size)
        chrom[idx] = 1 - chrom[idx]
    return offspring

# ---------- KRZYŻOWANIA (real) --------------------------------------------
def arithmetic_crossover(parents, offspring_size, ga_instance):
    alpha = 0.5
    offspring = np.empty(offspring_size, dtype=float)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k+1) % parents.shape[0]]
        offspring[k] = alpha*p1 + (1-alpha)*p2
    return offspring

# (Analogicznie możesz dopisać linear, blx_alpha, itp.)

# ---------- MUTACJE (real) -------------------------------------------------
def mutation_gauss(offspring, ga_instance):
    sigma = 0.1                                # <- tu ustalasz stałą lub
    for chrom in offspring:                    #    weź z ga_instance.user_data
        idx = np.random.randint(chrom.size)
        chrom[idx] += np.random.normal(0, sigma)
    return offspring