from pso import ParticleSwarmOptimization
from data_manager import DataManager
import matplotlib.pyplot as plt
import numpy as np
import random


class Hyperparams:
    def __init__(self):
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.cognitive_parameter = 0.6
        self.social_parameter = 0.8


random.seed(23)
hyperparams = Hyperparams()
data_manager = DataManager()
lower_bound = np.array([0.0, 0.0, 0.0])
upper_bound = np.array([1.0, 1.0, 1.0])


num_steps = 50
do_mont_carlo = True
num_samples = 50
samples_size = 12  # Number of months investing
do_plot = False


additional_args = [do_mont_carlo, num_samples, samples_size]
pso = ParticleSwarmOptimization(hyperparams, lower_bound, upper_bound)
pso.find_max(data_manager.calculate_ark_profit, num_steps, additional_args)

[cdi, ifix, ibov] = pso.best_global_position

if do_plot:
    data_manager.acc_profit.plot()
    plt.show()

print("Arca ideal: ")
print("CDI: ", cdi, "IFIX: ", ifix, "IBOV: ", ibov, "S&P500: ", 1 - (cdi + ifix + ibov))
print("Melhor lucro médio da arca ideal (", samples_size if do_mont_carlo else len(data_manager.monthly_profit),
      "meses): ", pso.best_global_value - 1)
print("\nLucros médios individuais(", samples_size if do_mont_carlo else len(data_manager.monthly_profit),
      "meses ):")
print("CDI          ", data_manager.calculate_ark_profit([1.0, 0.0, 0.0], additional_args) - 1)
print("IFIX         ", data_manager.calculate_ark_profit([0.0, 1.0, 0.0], additional_args) - 1)
print("IBOVESPA     ", data_manager.calculate_ark_profit([0.0, 0.0, 1.0], additional_args) - 1)
print("S&P500       ", data_manager.calculate_ark_profit([0.0, 0.0, 0.0], additional_args) - 1)

print("\nLucro médio arca 25% (", samples_size if do_mont_carlo else len(data_manager.monthly_profit),
      "meses):")
print(data_manager.calculate_ark_profit([0.25, 0.25, 0.25], additional_args) - 1)
