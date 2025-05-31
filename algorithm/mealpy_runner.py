from algorithm.custom_ao import CustomAO
from mealpy.utils.space import FloatVar
import numpy as np

class MealpyRunner:
    def __init__(self, func, minimize, lower_bound, upper_bound, num_variables, pop_size, epochs):
        self.func = func
        self.bounds = FloatVar(lb=[lower_bound] * num_variables, ub=[upper_bound] * num_variables)
        self.minimize = minimize
        self.pop_size = pop_size
        self.epochs = epochs

    def _fitness_wrapper(self, x):
        return self.func(x)

    def run(self):
        model = CustomAO(epoch=self.epochs, pop_size=self.pop_size)
        problem = {
            "obj_func": self._fitness_wrapper,
            "bounds": self.bounds,
            "minmax": "min" if self.minimize else "max"
        }
        g_best = model.solve(problem)

        list_best = model.history.list_global_best_fit
        avg_history = model._avg_history
        std_history = model._std_history

        return g_best.solution, g_best.target.fitness, list_best, avg_history, std_history

