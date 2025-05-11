import random
import numpy as np

def select_parents(population, fitnesses, method, tournament_size, minimize):
    # Metoda selekcji: wybiera 2 rodziców na podstawie wybranej strategii

    if method == "best":
        # Wybór 2 najlepszych osobników z całej populacji
        parents = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=not minimize)
        return [x[0] for x in parents[:2]]

    elif method == "roulette":
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            return random.sample(population, 2)
        probs = [f / total_fitness for f in fitnesses]
        indices = np.random.choice(len(population), size=2, replace=False, p=probs)
        return [population[i] for i in indices]

    elif method == "tournament":
        # Selekcja turniejowa - wybierz najlepszych z losowej podgrupy
        def tournament():
            # Losuj `tournament_size` osobników, wybierz najlepszego z nich
            competitors = random.sample(list(zip(population, fitnesses)), tournament_size)
            # Zwróć najlepszego osobnika, uwzględniając minimalizację lub maksymalizację
            if minimize:
                return min(competitors, key=lambda x: x[1])[0]  # dla minimalizacji
            else:
                return max(competitors, key=lambda x: x[1])[0]  # dla maksymalizacji
        return [tournament(), tournament()]  # dwukrotnie wybieramy rodzica
    return None
