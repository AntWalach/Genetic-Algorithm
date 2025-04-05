import random

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