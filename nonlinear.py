import numpy as np
from scipy.sparse import issparse

class HypergraphContagion:
    def __init__(self, incidence_matrix, initial_infected, lambda_val, nu, mu=0.015):
        self.incidence_matrix = incidence_matrix
        self.state = initial_infected.copy()
        self.lambda_val = lambda_val
        self.nu = nu
        self.mu = mu
        self.num_nodes = incidence_matrix.shape[0]
        self.num_edges = incidence_matrix.shape[1]

    def infection_probability(self, i):
        return min(self.lambda_val * (i ** self.nu), 1.0)

    def simulate(self, max_steps=1000, verbose=False, tol=1e-4):
        infected_fraction = np.mean(self.state)
        prev_infected = infected_fraction

        for step in range(max_steps):
            new_state = self.state.copy()

            # Recovery step
            recovery_probs = np.random.random(self.num_nodes)
            new_state[self.state == 1] = (recovery_probs[self.state == 1] > self.mu).astype(int)

            # Infection step
            for e in range(self.num_edges):
                if issparse(self.incidence_matrix):
                    nodes_e = self.incidence_matrix[:, e].nonzero()[0]
                else:
                    nodes_e = np.where(self.incidence_matrix[:, e])[0]
                if len(nodes_e) == 0:
                    continue

                infected_count = np.sum(self.state[nodes_e])
                if infected_count == 0 or infected_count == len(nodes_e):
                    continue

                prob = self.infection_probability(infected_count)
                susceptible_nodes = nodes_e[self.state[nodes_e] == 0]
                infection_probs = np.random.random(len(susceptible_nodes))
                new_state[susceptible_nodes] = (infection_probs < prob).astype(int)

            self.state = new_state
            infected_fraction = np.mean(self.state)

            if abs(infected_fraction - prev_infected) < tol:
                break
            prev_infected = infected_fraction

        return infected_fraction