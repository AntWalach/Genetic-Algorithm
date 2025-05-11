import numpy as np
from algorithm.selection import select_parents
from algorithm.crossover import (arithmetic_crossover, linear_crossover, blend_crossover_alpha, blend_crossover_alpha_beta, averaging_crossover)
from algorithm.mutation import uniform_mutation, gaussian_mutation
from algorithm.utils_real import initialize_real_population, clip_to_bounds

class RealGeneticAlgorithm:
    def __init__(self, func, minimize=True,
                 population_size=50, num_epochs=100,
                 selection_method="best", tournament_size=3,
                 crossover_method="arithmetic", crossover_prob=0.8,
                 mutation_method="gaussian", mutation_prob=0.05,
                 sigma=0.1,
                 elitism_rate=0.1,
                 lower_bound=-5, upper_bound=5,
                 num_variables=10):
        self.func = func
        self.minimize = minimize
        self.population_size = population_size
        self.num_epochs = num_epochs
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_method = crossover_method
        self.crossover_prob = crossover_prob
        self.mutation_method = mutation_method
        self.mutation_prob = mutation_prob
        self.sigma = sigma
        self.elitism_count = max(1, int(elitism_rate * population_size / 100))
        self.low, self.high = lower_bound, upper_bound
        self.num_variables = num_variables


    def _fitness(self, individual):
        return self.func(individual)

    def _do_crossover(self, p1, p2):
        if np.random.rand() > self.crossover_prob:
            return p1.copy(), p2.copy()

        if self.crossover_method == "arithmetic":
            return arithmetic_crossover(p1, p2)
        if self.crossover_method == "linear":
            kids = linear_crossover(p1, p2)
            fit = [self._fitness(k) for k in kids]
            idx = np.argsort(fit) if self.minimize else np.argsort(fit)[::-1]
            return kids[idx[0]], kids[idx[1]]
        if self.crossover_method == "blend_crossover_alpha":
            return blend_crossover_alpha(p1, p2), blend_crossover_alpha(p2, p1)
        if self.crossover_method == "blend_crossover_alpha_beta":
            return blend_crossover_alpha_beta(p1, p2), blend_crossover_alpha_beta(p2, p1)
        if self.crossover_method == "averaging":
            child = averaging_crossover(p1, p2)
            return child, child.copy()
        raise ValueError("Unknown crossover method")

    def _do_mutation(self, child):
        if self.mutation_method == "uniform":
            child = uniform_mutation(child, self.mutation_prob,
                                     self.low, self.high)
        elif self.mutation_method == "gaussian":
            child = gaussian_mutation(child, self.mutation_prob,
                                      self.sigma, self.low, self.high)
        return child


    def run(self, return_statistics=False):
        pop = initialize_real_population(self.population_size,
                                         self.num_variables,
                                         self.low, self.high)
        best, best_fit = None, np.inf if self.minimize else -np.inf
        history, avg_history, std_history = [], [], []

        for _ in range(self.num_epochs):
            fitnesses = [self._fitness(ind) for ind in pop]

            # zapis statystyk
            current_best_idx = int(np.argmin(fitnesses) if self.minimize else np.argmax(fitnesses))
            if self.minimize and fitnesses[current_best_idx] < best_fit or \
               not self.minimize and fitnesses[current_best_idx] > best_fit:
                best_fit = fitnesses[current_best_idx]
                best = pop[current_best_idx].copy()
            history.append(best_fit)
            avg_history.append(float(np.mean(fitnesses)))
            std_history.append(float(np.std(fitnesses)))

            # elita
            elite_idx = np.argsort(fitnesses) if self.minimize else np.argsort(fitnesses)[::-1]
            new_pop = [pop[i].copy() for i in elite_idx[:self.elitism_count]]

            # reszta populacji
            while len(new_pop) < self.population_size:
                p1, p2 = select_parents(pop, fitnesses,
                                        self.selection_method,
                                        self.tournament_size,
                                        self.minimize)
                c1, c2 = self._do_crossover(p1, p2)
                c1 = self._do_mutation(c1)
                c2 = self._do_mutation(c2)
                new_pop.extend([clip_to_bounds(c1, self.low, self.high),
                                clip_to_bounds(c2, self.low, self.high)])
            pop = new_pop[:self.population_size]

        if return_statistics:
            return best, best_fit, history, avg_history, std_history
        return best, best_fit
