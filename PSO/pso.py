import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """

    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        delta = upper_bound - lower_bound
        self.position = np.random.uniform(lower_bound, upper_bound)
        self.velocity = np.random.uniform(-delta, delta)
        self.best_position = None
        self.best_value = -inf


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """

    def __init__(self, hyperparams, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.delta = upper_bound - lower_bound
        self.swarm_size = hyperparams.num_particles
        self.inertia_weight = hyperparams.inertia_weight
        self.cognitive_parameter = hyperparams.cognitive_parameter
        self.social_parameter = hyperparams.social_parameter

        self.particles = [Particle(lower_bound, upper_bound) for _ in range(self.swarm_size)]
        self.actual_index = 0
        self.actual_particle = self.particles[0]
        self.best_global_position = None
        self.best_global_value = -inf

    def update_particles(self):
        """
        Advances the generation of particles.
        """

        for particle in self.particles:
            rp = random.uniform(0.0, 1.0)
            rg = random.uniform(0.0, 1.0)

            particle.velocity = self.inertia_weight * particle.velocity + \
                                self.cognitive_parameter * rp * (particle.best_position - particle.position) + \
                                self.social_parameter * rg * (self.best_global_position - particle.position)
            for i in range(len(self.lower_bound)):
                # restrict velocity:
                if particle.velocity[i] > self.delta[i]:
                    particle.velocity[i] = self.delta[i]
                elif particle.velocity[i] < -self.delta[i]:
                    particle.velocity[i] = -self.delta[i]

            particle.position = particle.position + particle.velocity
            for i in range(len(self.lower_bound)):
                # restrict position:
                if particle.position[i] > self.upper_bound[i]:
                    particle.position[i] = self.upper_bound[i]
                    particle.velocity[i] = -particle.velocity[i]
                elif particle.position[i] < self.lower_bound[i]:
                    particle.position[i] = self.lower_bound[i]
                    particle.velocity[i] = -particle.velocity[i]

    def evaluate_particles(self, function):
        for particle in self.particles:
            value = function(particle.position)

            if value > particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position

            if value > self.best_global_value:
                self.best_global_value = value
                self.best_global_position = particle.position

    def find_max(self, function, num_steps):
        for _ in range(num_steps):
            self.evaluate_particles(function)
            self.update_particles()
