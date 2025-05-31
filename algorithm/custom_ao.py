import numpy as np
from mealpy.swarm_based.AO import OriginalAO


class CustomAO(OriginalAO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._avg_history = []
        self._std_history = []

    def evolve(self, epoch):
        super().evolve(epoch)

        fitnesses = [agent.target.fitness for agent in self.pop]
        self._avg_history.append(np.mean(fitnesses))
        self._std_history.append(np.std(fitnesses))
