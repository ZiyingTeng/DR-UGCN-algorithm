import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from UniGGCN import UniGCNRegression
from utils import load_hypergraph, compute_features, split_dataset, compute_infected_fraction, get_max_hyperdegree
from baseline import compute_dc, compute_bc, compute_hdc, compute_sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def train_model(model, data, train_idx, val_idx, epochs=400, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # Pairwise ranking loss
        pairs = torch.combinations(train_idx, r=2)
        diff = out[pairs[:, 0]] - out[pairs[:, 1]]
        target_diff = data.y[pairs[:, 0]] - data.y[pairs[:, 1]]
        loss = F.mse_loss(diff, target_diff)
        loss.backward()

        # Debug gradient norm
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()

        if epoch == epochs - 1:
            print(f"Final Epoch {epoch}/{epochs} ({100 * epoch / epochs:.1f}%), Grad Norm: {grad_norm:.4f}")

    # Check for missing gradients
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f'Warning: No gradient for parameter {name}')


def evaluate_algorithms(incidence_matrix, features, edge_index, top_k, lambda_vals, nu_vals):
    num_nodes = incidence_matrix.shape[0]
    Mmax = get_max_hyperdegree(incidence_matrix)
    edge_sizes = np.sum(incidence_matrix, axis=0).A1
    max_edge_size = np.max(edge_sizes)

    model = UniGCNRegression(
        in_channels=features.shape[1],
        hidden_channels=256,  # Increased for better capacity
        out_channels=1,
        num_layers=4,  # Increased for better capacity
        dropout=0.5
    )

    hdc = compute_hdc(incidence_matrix)
    scaler = MinMaxScaler()
    y = scaler.fit_transform(hdc.reshape(-1, 1))
    y = torch.tensor(y, dtype=torch.float)

    data = Data(x=features, edge_index=edge_index, y=y, Pv=None, PvT=None)
    data = model.unig_encoder.d_expansion(data)

    train_idx, val_idx, test_idx = split_dataset(num_nodes)

    train_model(model, data, train_idx, val_idx)

    model.eval()
    with torch.no_grad():
        scores_dr_ugcn = model(data).squeeze().cpu().numpy()

    scores_dc = compute_dc(incidence_matrix)
    scores_bc = compute_bc(incidence_matrix)
    scores_hdc = compute_hdc(incidence_matrix)
    scores_sc = compute_sc(incidence_matrix)

    results = {alg: [] for alg in ['DR-UGCN', 'DC', 'BC', 'HDC', 'SC']}
    for nu in nu_vals:
        lambda_val = 0.01  # Maintained for 0.65–0.95 range
        print(f"\nnu={nu:.2f}, lambda_val={lambda_val:.6f}:")
        for alg, scores in [('DR-UGCN', scores_dr_ugcn), ('DC', scores_dc),
                            ('BC', scores_bc), ('HDC', scores_hdc), ('SC', scores_sc)]:
            top_nodes = np.argsort(scores)[-top_k:]
            infected_frac = compute_infected_fraction(incidence_matrix, top_nodes, lambda_val, nu)
            results[alg].append(infected_frac)
            print(f"  {alg}: Infected fraction = {infected_frac:.4f}")

    return results, max_edge_size


def plot_results(nu_vals, results, Mmax, max_edge_size):
    plt.figure(figsize=(10, 6))
    colors = {'DR-UGCN': 'blue', 'DC': 'green', 'BC': 'red', 'HDC': 'purple', 'SC': 'orange'}

    for alg, fractions in results.items():
        plt.plot(nu_vals, fractions, label=alg, color=colors[alg], marker='o')

    plt.xlabel('Nonlinearity Degree (ν)')
    plt.ylabel('Infected Fraction')
    plt.title('Senate')
    plt.ylim(0, 1)  # Fixed y-axis range
    plt.text(1.1, 0.9, f"Max Node Hyperdegree: {Mmax}\nMax Hyperedge Size: {max_edge_size}")
    plt.legend()
    plt.grid(True)
    plt.savefig('infected_fraction_plot.png')
    plt.show()


if __name__ == "__main__":
    file_path = "hyperedges-senate-committees.txt"
    incidence_matrix, edge_index, node_id_map = load_hypergraph(file_path)
    features = compute_features(incidence_matrix)

    num_nodes = incidence_matrix.shape[0]
    top_k = int(0.1 * num_nodes)
    nu_vals = np.linspace(1.1, 1.9, 9)
    Mmax = get_max_hyperdegree(incidence_matrix)

    results, max_edge_size = evaluate_algorithms(incidence_matrix, features, edge_index, top_k, None, nu_vals)

    plot_results(nu_vals, results, Mmax, max_edge_size)