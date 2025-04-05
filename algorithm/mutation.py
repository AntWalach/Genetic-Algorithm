import random

def mutate(chromosome, method, prob, num_variables=None):
    # Mutacja chromosomu: w zależności od metody losowo zmienia bity z prawdopodobieństwem `prob`
    chromo = list(chromosome)

    if method == "bit_flip":
        # Mutacja bitowa: każdy bit może się zmienić niezależnie z prawdopodobieństwem `prob`
        for i in range(len(chromo)):
            if random.random() < prob:
                chromo[i] = '1' if chromo[i] == '0' else '0'

    elif method == "one_point":
        # Mutacja jednopunktowa: losowy jeden bit zmienia się z prawdopodobieństwem `prob`
        if random.random() < prob:
            index = random.randint(0, len(chromo) - 1)
            chromo[index] = '1' if chromo[index] == '0' else '0'

    elif method == "two_point":
        # Mutacja dwupunktowa: odwraca bity między dwoma losowymi punktami (jeśli zajdzie mutacja)
        if random.random() < prob:
            i, j = sorted(random.sample(range(len(chromo)), 2))
            for k in range(i, j):
                chromo[k] = '1' if chromo[k] == '0' else '0'

    elif method == "boundary" and num_variables is not None:
        # Mutacja brzegowa: przypisuje jednej losowej zmiennej wartość graniczną (0 lub 1)
        if random.random() < prob:
            bits_per_var = len(chromo) // num_variables
            var_index = random.randint(0, num_variables - 1)
            start = var_index * bits_per_var
            end = start + bits_per_var
            value = random.choice(['0', '1'])
            chromo[start:end] = [value] * bits_per_var

    return ''.join(chromo)
