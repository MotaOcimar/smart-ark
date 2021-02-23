from pso import ParticleSwarmOptimization


class Hyperparams:
    def __init__(self):
        self.num_particles = 50
        self.inertia_weight = 0.9
        self.cognitive_parameter = 0.6
        self.social_parameter = 0.8


hyperparams = Hyperparams()
lower_bound = 0.0
upper_bound = 1.0
num_steps = 100

pso = ParticleSwarmOptimization(hyperparams, lower_bound, upper_bound)
pso.find_max(hyperparams, num_steps)

[cdi, ifix, sp500, ibov] = pso.best_global_position
