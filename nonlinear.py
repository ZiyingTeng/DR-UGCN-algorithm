# import numpy as np
# from tqdm import tqdm
#
# class HypergraphContagion:
#     def __init__(self, hypergraph_incidence, initial_infected, lambda_val, nu, mu=0.5):
#         self.H = hypergraph_incidence
#         self.N, self.E = self.H.shape
#         self.lambda_val = lambda_val
#         self.nu = nu
#         self.mu = mu
#         self.infected = initial_infected.copy()
#         self.susceptible = 1 - self.infected
#         self.edge_infected_counts = np.zeros(self.E, dtype=int)
#         self.update_edge_infected_counts()
#         self.node_infection_rates = np.zeros(self.N)
#         self.update_node_infection_rates()
#
#     def update_edge_infected_counts(self):
#         self.edge_infected_counts = (self.H.T @ self.infected).astype(int)
#
#     def update_node_infection_rates(self):
#         edge_rates = self.lambda_val * (1 + self.edge_infected_counts) ** self.nu
#         self.node_infection_rates = (self.H @ edge_rates) * self.susceptible
#
#     def step(self):
#         total_infection_rate = np.sum(self.node_infection_rates)
#         total_recovery_rate = np.sum(self.infected) * self.mu
#         total_rate = total_infection_rate + total_recovery_rate
#         if total_rate == 0:
#             return False
#         if np.random.rand() < total_infection_rate / total_rate:
#             probs = self.node_infection_rates / total_infection_rate
#             node = np.random.choice(self.N, p=probs)
#             self.infected[node] = 1
#             self.susceptible[node] = 0
#         else:
#             probs = self.infected / np.sum(self.infected)
#             node = np.random.choice(self.N, p=probs)
#             self.infected[node] = 0
#             self.susceptible[node] = 1
#         self.update_edge_infected_counts()
#         self.update_node_infection_rates()
#         return True
#
#     def simulate(self, max_steps=10000, tolerance=1e-5, verbose=False):
#         prev_infected = np.sum(self.infected)
#         stationary_steps = 0
#         iterator = tqdm(range(max_steps)) if verbose else range(max_steps)
#         for step in iterator:
#             if not self.step():
#                 break
#             current_infected = np.sum(self.infected)
#             if abs(current_infected - prev_infected) < tolerance:
#                 stationary_steps += 1
#                 if stationary_steps >= 500:
#                     break
#             else:
#                 stationary_steps = 0
#             prev_infected = current_infected
#         return np.mean(self.infected)


# 跑txt用的


import numpy as np
from scipy.special import comb
from tqdm import tqdm

class HypergraphContagion:
    def __init__(self, hypergraph_incidence, initial_infected, lambda_val, nu, mu=1.0, beta=0.5):
        self.H = hypergraph_incidence
        self.N, self.E = self.H.shape
        self.lambda_val = lambda_val
        self.nu = nu
        self.mu = mu
        self.beta = beta
        self.infected = initial_infected.copy()
        self.susceptible = 1 - self.infected
        self.node_degrees = np.sum(self.H, axis=1).A1.astype(float)
        self.node_degrees = self.node_degrees / (np.max(self.node_degrees) + 1e-10)
        self.edge_infected_counts = np.zeros(self.E, dtype=int)
        self.update_edge_infected_counts()
        self.node_infection_rates = np.zeros(self.N)
        self.update_node_infection_rates()

    def update_edge_infected_counts(self):
        self.edge_infected_counts = (self.H.T @ self.infected).astype(int)
        if np.any(self.edge_infected_counts < 0):
            raise ValueError("Negative edge infected counts detected")

    def update_node_infection_rates(self):
        edge_rates = self.lambda_val * (1 + self.edge_infected_counts) ** self.nu
        self.node_infection_rates = (self.H @ edge_rates) * self.susceptible
        self.node_infection_rates *= self.node_degrees ** self.beta

    def step(self):
        total_infection_rate = np.sum(self.node_infection_rates)
        total_recovery_rate = np.sum(self.infected) * self.mu
        total_rate = total_infection_rate + total_recovery_rate
        min_rate = max(1e-3, self.lambda_val * 1e-2)
        if total_rate < min_rate or total_infection_rate == 0:
            return False
        if np.random.rand() < total_infection_rate / total_rate:
            probs = self.node_infection_rates / (total_infection_rate + 1e-10)
            node = np.random.choice(self.N, p=probs)
            self.infected[node] = 1
            self.susceptible[node] = 0
        else:
            probs = self.infected / (np.sum(self.infected) + 1e-10)
            node = np.random.choice(self.N, p=probs)
            self.infected[node] = 0
            self.susceptible[node] = 1
        self.update_edge_infected_counts()
        self.update_node_infection_rates()
        return True

    def simulate(self, max_steps=15000, tolerance=1e-6, min_rate=1e-4, stable_steps=200):
        min_rate = max(min_rate, self.lambda_val * 1e-2)
        prev_infected = np.sum(self.infected)
        stationary_steps = 0
        rate_low_steps = 0
        infected_fractions = []

        for step in range(max_steps):
            if not self.step():
                rate_low_steps += 1
                if rate_low_steps >= stable_steps:
                    break
            else:
                rate_low_steps = 0

            current_infected = np.sum(self.infected)
            infected_fractions.append(current_infected / self.N)

            if abs(current_infected - prev_infected) < tolerance:
                stationary_steps += 1
                if stationary_steps >= stable_steps:
                    break
            else:
                stationary_steps = 0

            prev_infected = current_infected

        return np.mean(infected_fractions[-100:]) if len(infected_fractions) >= 100 else np.mean(infected_fractions)


