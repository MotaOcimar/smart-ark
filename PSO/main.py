from pso import ParticleSwarmOptimization
from data_manager import DataManager
import numpy as np


class Hyperparams:
    def __init__(self):
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.cognitive_parameter = 0.6
        self.social_parameter = 0.8


hyperparams = Hyperparams()
data_manager = DataManager()
lower_bound = np.array([0.0, 0.0, 0.0])
upper_bound = np.array([1.0, 1.0, 1.0])
num_steps = 100

pso = ParticleSwarmOptimization(hyperparams, lower_bound, upper_bound)
pso.find_max(data_manager.calculate_total_profit, num_steps)

[cdi, ifix, ibov] = pso.best_global_position

print(cdi, ifix, ibov, 1 - (cdi + ifix + ibov))
