import random
import numpy as np
def crossover(p1, p2, method, prob):
    # Zwraca dwójkę potomków po krzyżowaniu p1 i p2 według wybranej metody
    # Z szansą `prob` wykonuje krzyżowanie, w przeciwnym razie zwraca oryginalne rodzicielskie chromosomy

    if random.random() > prob:
        return p1, p2  # brak krzyżowania - zwracamy bez zmian

    if method == "one_point":
        # Krzyżowanie jednopunktowe
        point = random.randint(1, len(p1)-1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    elif method == "two_point":
        # Krzyżowanie dwupunktowe
        point1 = random.randint(1, len(p1) - 2)
        point2 = random.randint(point1 + 1, len(p1) - 1)
        return (
            p1[:point1] + p2[point1:point2] + p1[point2:],
            p2[:point1] + p1[point1:point2] + p2[point2:]
        )

    elif method == "uniform":
        # Krzyżowanie jednorodne - losowo wybieramy bity z p1 lub p2 (50/50)
        return (
            ''.join(p1[i] if random.random() < 0.5 else p2[i] for i in range(len(p1))),
            ''.join(p2[i] if random.random() < 0.5 else p1[i] for i in range(len(p2)))
        )

    elif method == "granular":
        # Krzyżowanie ziarniste - dla każdego bitu losowo wybieramy p1[i] lub p2[i]
        return (
            ''.join(random.choice([p1[i], p2[i]]) for i in range(len(p1))),
            ''.join(random.choice([p1[i], p2[i]]) for i in range(len(p2)))
        )

def arithmetic_crossover(p1, p2, alpha=0.5):
    return alpha * p1 + (1 - alpha) * p2, (1 - alpha) * p1 + alpha * p2

def linear_crossover(p1, p2):
    c1 = 0.5 * p1 + 0.5 * p2
    c2 = 1.5 * p1 - 0.5 * p2
    c3 = -0.5 * p1 + 1.5 * p2
    return c1, c2, c3

def blend_crossover_alpha(p1, p2, alpha=0.3):
    low  = np.minimum(p1, p2)
    high = np.maximum(p1, p2)
    diff = high - low
    return np.random.uniform(low - alpha*diff, high + alpha*diff)

def blend_crossover_alpha_beta(p1, p2, alpha=0.5, beta=0.5):
    child = np.empty_like(p1)
    for i, (x, y) in enumerate(zip(p1, p2)):
        d = abs(x - y)
        low  = min(x, y) - alpha * d
        high = max(x, y) + beta  * d
        child[i] = np.random.uniform(low, high)
    return child

def averaging_crossover(p1, p2):
    return (p1 + p2) / 2.0
