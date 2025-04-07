import numpy as np
from algorithm.selection import select_parents
from algorithm.crossover import crossover
from algorithm.mutation import mutate
from algorithm.inversion import apply_inversion
from algorithm.utils import calculate_chromosome_length, decode, generate_chromosome, initialize_population

class GeneticAlgorithm:
    def __init__(self, func, minimize=True, precision=0.01,
                 population_size=50, num_epochs=100,
                 selection_method="best", tournament_size=3,
                 crossover_method="one_point", crossover_prob=0.8,
                 mutation_method="one_point", mutation_prob=0.05,
                 inversion_prob=0.1,
                 elitism_rate=0.1,
                 lower_bound=-5,
                 upper_bound=5):
        # Inicjalizacja parametrów algorytmu genetycznego
        self.func = func  # funkcja celu (fitness)
        self.minimize = minimize  # True = minimalizacja, False = maksymalizacja
        self.precision = precision  # dokładność kodowania zmiennych
        self.population_size = population_size  # liczba osobników
        self.num_epochs = num_epochs  # liczba epok (generacji)

        self.selection_method = selection_method  # metoda selekcji
        self.tournament_size = tournament_size  # rozmiar turnieju

        self.crossover_method = crossover_method  # metoda krzyżowania
        self.crossover_prob = crossover_prob  # prawdopodobieństwo krzyżowania

        self.mutation_method = mutation_method  # metoda mutacji
        self.mutation_prob = mutation_prob  # prawdopodobieństwo mutacji

        self.inversion_prob = inversion_prob  # prawdopodobieństwo inwersji
        self.elitism_count = max(1, int(elitism_rate * population_size / 100))  # liczba elitarnych osobników

        self.lower_bound = lower_bound  # dolna granica zmiennych
        self.upper_bound = upper_bound  # górna granica zmiennych

        self.num_variables = None  # liczba zmiennych (ustawiana później)
        self.chromosome_length = None  # długość całkowita chromosomu

    def fitness(self, chromosome):
        # Obliczenie wartości funkcji celu na podstawie zakodowanego chromosomu
        x = decode(chromosome, self.chromosome_length, self.num_variables, self.lower_bound, self.upper_bound)
        return self.func(x)

    def run(self, return_statistics=False):
        # Główna pętla ewolucji
        self.chromosome_length = calculate_chromosome_length(
            self.lower_bound, self.upper_bound, self.precision, self.num_variables)

        population = initialize_population(self.population_size, self.chromosome_length)
        best_solution = None
        best_fitness = float('-inf') if not self.minimize else float('inf')
        history, avg_history, std_history = [], [], []

        for epoch in range(self.num_epochs):
            # Obliczanie fitnessów
            fitnesses = [self.fitness(ind) for ind in population]

            # Wybór najlepszego osobnika w tej epoce
            current_best = min(zip(population, fitnesses), key=lambda x: x[1]) if self.minimize else \
                            max(zip(population, fitnesses), key=lambda x: x[1])

            # Aktualizacja najlepszego globalnego rozwiązania
            if (self.minimize and current_best[1] < best_fitness) or \
               (not self.minimize and current_best[1] > best_fitness):
                best_fitness = current_best[1]
                best_solution = current_best[0]

            # Statystyki
            history.append(best_fitness)
            avg_history.append(np.mean(fitnesses))
            std_history.append(np.std(fitnesses))

            # Selekcja elity
            elite = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=not self.minimize)[:self.elitism_count]
            new_population = [x[0] for x in elite]

            # Tworzenie reszty populacji
            while len(new_population) < self.population_size:
                p1, p2 = select_parents(population, fitnesses, self.selection_method, self.tournament_size)
                c1, c2 = crossover(p1, p2, self.crossover_method, self.crossover_prob)
                c1 = mutate(c1, self.mutation_method, self.mutation_prob, self.num_variables)
                c2 = mutate(c2, self.mutation_method, self.mutation_prob, self.num_variables)
                c1 = apply_inversion(c1, self.inversion_prob)
                c2 = apply_inversion(c2, self.inversion_prob)
                new_population.extend([c1, c2])

            population = new_population[:self.population_size]  # obcięcie do populacji docelowej

        # Dekodowanie najlepszego rozwiązania
        best_decoded = decode(best_solution, self.chromosome_length, self.num_variables, self.lower_bound, self.upper_bound)
        print("\nKONIEC")
        print("Populacja:")
        fitnesses = [self.fitness(ind) for ind in population]

        for ind, fit in zip(population, fitnesses):
            print(f"Individual: {ind} | target func: {fit:.4f}")

        # WYPISYWANIE TOP 5 OSOBNIKÓW
        print("\nNajlepsi osobnicy:")
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1])
        for i, (ind, fit) in enumerate(sorted_pop[:5]):
            print(f"{i + 1}. {ind} | target func: {fit:.4f}")
        return (best_decoded, best_fitness, history, avg_history, std_history) if return_statistics else (best_decoded, best_fitness, history)
