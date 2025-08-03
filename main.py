import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from UniGGCN import UniGCNRegression
from utils import load_hypergraph, compute_features, split_dataset, compute_infected_fraction, get_max_hyperdegree, \
    compute_simulation_labels
from baseline import compute_dc, compute_bc, compute_hdc, compute_sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model, data, train_idx, val_idx, incidence_matrix, lambda_val=0.01, nu=1.9, epochs=400, lr=0.005,
                patience=150):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    best_loss = float('inf')
    early_stop_counter = 0

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # Combined loss: pairwise ranking + MSE + simulation-based
        pairs = torch.combinations(train_idx, r=2)
        diff = out[pairs[:, 0]] - out[pairs[:, 1]]
        target_diff = data.y[pairs[:, 0]] - data.y[pairs[:, 1]]
        ranking_loss = F.mse_loss(torch.clamp(diff - target_diff + 0.5, min=0), torch.zeros_like(diff))  # Margin to 0.5
        mse_loss = F.mse_loss(out[train_idx], data.y[train_idx])
        # Simulation-based loss: maximize infected fraction for top-k
        k = int(0.1 * len(train_idx))
        top_k_indices = torch.topk(out[train_idx], k, dim=0)[1]
        top_k_nodes = train_idx[top_k_indices].cpu().numpy()
        infected_frac = compute_infected_fraction(incidence_matrix, top_k_nodes, lambda_val, nu)
        sim_loss = -infected_frac  # Negative to maximize
        loss = 0.6 * ranking_loss + 0.2 * mse_loss + 0.2 * sim_loss  # Adjusted weights
        loss.backward()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch == epochs - 1 or early_stop_counter >= patience:
            print(
                f"Final Epoch {epoch}/{epochs} ({100 * epoch / epochs:.1f}%), Grad Norm: {grad_norm:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

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
        hidden_channels=768,
        out_channels=1,
        num_layers=4,
        dropout=0.3,
        batch_norm=True
    )

    # Simulation-guided labels
    sim_labels = compute_simulation_labels(incidence_matrix, lambda_val=0.01, nu=1.9)
    scaler = MinMaxScaler()
    y = torch.tensor(scaler.fit_transform(sim_labels.reshape(-1, 1)), dtype=torch.float)

    data = Data(x=features, edge_index=edge_index, y=y, Pv=None, PvT=None)
    data = model.unig_encoder.d_expansion(data)

    train_idx, val_idx, test_idx = split_dataset(num_nodes)

    train_model(model, data, train_idx, val_idx, incidence_matrix)

    model.eval()
    with torch.no_grad():
        scores_dr_ugcn = model(data).squeeze().cpu()
        # Hybrid top-k selection: softmax + ranking cutoff
        scores_softmax = torch.softmax(scores_dr_ugcn / 0.1, dim=0).numpy()  # Temperature to 0.1
        scores_rank = np.argsort(scores_dr_ugcn)[-top_k:]  # Top-k indices
        scores_dr_ugcn = np.zeros_like(scores_dr_ugcn)
        scores_dr_ugcn[scores_rank] = scores_softmax[scores_rank]

    # Debug top-k nodes
    top_nodes = np.argsort(scores_dr_ugcn)[-top_k:]
    hyperdegrees = np.sum(incidence_matrix[top_nodes], axis=1).A1
    hyperedge_sizes = [
        np.sum(incidence_matrix[:, np.where(incidence_matrix[top_nodes[i]].toarray().flatten())[0]], axis=0).A1 for i in
        range(top_k)]
    print(f"DR-UGCN Top-k Nodes: {top_nodes}")
    print(f"Hyperdegrees: {hyperdegrees}")
    print(f"Hyperedge sizes for top-k nodes: {hyperedge_sizes}")
    top_hdc = np.argsort(compute_hdc(incidence_matrix))[-top_k:]
    hdc_hyperdegrees = np.sum(incidence_matrix[top_hdc], axis=1).A1
    hdc_hyperedge_sizes = [
        np.sum(incidence_matrix[:, np.where(incidence_matrix[top_hdc[i]].toarray().flatten())[0]], axis=0).A1 for i in
        range(top_k)]
    print(f"HDC Top-k Nodes: {top_hdc}")
    print(f"HDC Hyperdegrees: {hdc_hyperdegrees}")
    print(f"HDC Hyperedge sizes for top-k nodes: {hdc_hyperedge_sizes}")

    scores_dc = compute_dc(incidence_matrix)
    scores_bc = compute_bc(incidence_matrix)
    scores_hdc = compute_hdc(incidence_matrix)
    scores_sc = compute_sc(incidence_matrix)

    results = {alg: [] for alg in ['DR-UGCN', 'DC', 'BC', 'HDC', 'SC']}
    for nu in nu_vals:
        lambda_val = 0.01
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
    plt.ylim(0, 1)
    plt.text(1.1, 0.9, f"Max Node Hyperdegree: {Mmax}\nMax Hyperedge Size: {max_edge_size}")
    plt.legend()
    plt.grid(True)
    plt.savefig('infected_fraction_plot.png')
    plt.show()


if __name__ == "__main__":
    file_path = "hyperedges-senate-committees.txt"  # Update for other datasets
    incidence_matrix, edge_index, node_id_map = load_hypergraph(file_path)
    features = compute_features(incidence_matrix)

    num_nodes = incidence_matrix.shape[0]
    top_k = int(0.1 * num_nodes)
    nu_vals = np.linspace(1.1, 1.9, 9)
    Mmax = get_max_hyperdegree(incidence_matrix)

    results, max_edge_size = evaluate_algorithms(incidence_matrix, features, edge_index, top_k, None, nu_vals)

    plot_results(nu_vals, results, Mmax, max_edge_size)