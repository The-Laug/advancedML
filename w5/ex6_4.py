import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def parameterize_curve(start, end, num_points):
    # Create a linear space of points between start and end
    t = torch.linspace(0, 1, num_points).unsqueeze(1)  # Shape: (num_points, 1)
    curve = start + t * (end - start)  # Linear interpolation (num_points, 2)

    # We only want the interior points to be trainable parameters. Create
    # an nn.Parameter for the interior points and return it alongside a
    # helper that can assemble the full curve for optimization.
    if num_points <= 2:
        # No interior points to optimize
        return curve, None

    interior_init = curve[1:-1].clone().detach()
    interior_param = nn.Parameter(interior_init)
    return curve, interior_param

#Generalized metric for batch size B
def metric(x_batch):
    # x_batch shape: (B, 2) where B is the number of midpoints
    B = x_batch.shape[0]
    
    # Calculate the scalar part: (1 + ||x||^2) for each point. Shape: (B,)
    scalars = 1.0 + torch.sum(x_batch**2, dim=1)
    
    # Create a batch of 2x2 identity matrices. Shape: (B, 2, 2)
    I = torch.eye(2, dtype=x_batch.dtype, device=x_batch.device).unsqueeze(0).expand(B, 2, 2)
    
    # Reshape scalars to (B, 1, 1) so it broadcasts correctly across the matrices
    G = scalars.view(B, 1, 1) * I
    
    # G now contains a 2x2 matrix for every single point in the batch!
    return G

def calc_energy(curve, metric=metric):
    """
    Calculates the energy of a curve based on a generalized metric.

    Args:
        curve (torch.Tensor): Tensor of shape (N, 2) representing N points in 2D space.

    Returns:
        torch.Tensor: Scalar tensor representing the total energy of the curve.
    """
    # 1. Calculate the velocity vectors (Delta x). Shape: (B, 2)
    deriv = curve[1:] - curve[:-1] 
    
    # 2. Calculate the midpoints. Shape: (B, 2)
    midpoints = (curve[1:] + curve[:-1]) / 2.0 
    
    # 3. Get the batch of metric matrices at the midpoints. Shape: (B, 2, 2)
    G = metric(midpoints) 
    
    # 4. Compute Delta x^T * G * Delta x without einsum
    # First, make 'deriv' a column vector so we can multiply it by G.
    # unsqueeze(-1) changes shape from (B, 2) to (B, 2, 1)
    deriv_col = deriv.unsqueeze(-1)
    
    # Multiply the matrix G by the column vector. Shape becomes (B, 2, 1)
    G_v = torch.matmul(G, deriv_col) 
    
    # Squeeze it back to a standard batched vector of shape (B, 2)
    G_v = G_v.squeeze(-1) 
    
    # Finally, multiply by Delta x^T. For vectors, v^T * (G*v) is just the dot product.
    # We do this via element-wise multiplication and summing across the coordinates (dim=1)
    segment_energies = torch.sum(deriv * G_v, dim=1) # Shape: (B,)
    
    # 5. Sum up the energy of all segments
    total_energy = torch.sum(segment_energies)
    
    return total_energy


def train_curve(start, end, num_points, num_iterations, learning_rate):
    base_curve, interior = parameterize_curve(start, end, num_points)

    # If there are interior parameters, optimize them; otherwise nothing to do
    if interior is not None:
        optimizer = torch.optim.Adam([interior], lr=learning_rate)
    else:
        optimizer = None

    for iteration in range(num_iterations):
        if optimizer is not None:
            optimizer.zero_grad()

        # assemble full curve: start, interior (if any), end
        if interior is None:
            curve = base_curve
        else:
            curve = torch.vstack((start.unsqueeze(0), interior, end.unsqueeze(0)))

        energy = calc_energy(curve)
        # if there's nothing to optimize, just compute and break
        if optimizer is None:
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Energy: {energy.item()}")
            continue

        energy.backward()
        optimizer.step()

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Energy: {energy.item()}")

    # return the final assembled curve
    if interior is None:
        return base_curve
    else:
        return torch.vstack((start.unsqueeze(0), interior.detach(), end.unsqueeze(0)))


def plot_optimization_results(start_tensor, end_tensor, optimized_tensor, metric_fn=metric):
    """
    Plots the original straight line, the optimized curve, and the 
    energy landscape of the metric space.
    """
    # Convert PyTorch tensors to NumPy arrays for plotting
    orig_x = [start_tensor[0].item(), end_tensor[0].item()]
    orig_y = [start_tensor[1].item(), end_tensor[1].item()]
    
    opt_x = optimized_tensor[:, 0].detach().cpu().numpy()
    opt_y = optimized_tensor[:, 1].detach().cpu().numpy()
    
    plt.figure(figsize=(8, 8))

    #bounds
    x_max = max(start_tensor[0].item(), end_tensor[0].item())
    y_max = max(start_tensor[1].item(), end_tensor[1].item())
    x_min = min(start_tensor[0].item(), end_tensor[0].item())
    y_min = min(start_tensor[1].item(), end_tensor[1].item())

    sym_max = max(x_max, y_max)
    sym_min = min(x_min, y_min)

    # 1. Plot the energy landscape (the scalar part of your metric)
    # We create a grid slightly larger than the [0, 1] bounds
    grid_x, grid_y = np.meshgrid(np.linspace(sym_min - 0.2, sym_max + 0.2, 100), 
                                 np.linspace(sym_min - 0.2, sym_max + 0.2, 100))
    # Calculate 1 + x^2 + y^2
    grid_z = 1.0 + grid_x**2 + grid_y**2 
    
    # Use a contour plot to show the "cost" of traveling through that region
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis_r', alpha=0.5)
    plt.colorbar(contour, label='Metric Cost (1 + x² + y²)')
    
    # 2. Plot the original straight path
    plt.plot(orig_x, orig_y, 'k--', label='Euclidean Straight Line', linewidth=2)
    
    # 3. Plot the optimized path
    plt.plot(opt_x, opt_y, 'r-o', label='Optimized Curve (Geodesic)', linewidth=2, markersize=6)
    
    # Highlight the start and end points
    plt.plot(orig_x[0], orig_y[0], 'go', label='Start', markersize=8, zorder=5)
    plt.plot(orig_x[1], orig_y[1], 'bo', label='End', markersize=8, zorder=5)
    
    
    # 4. Final formatting touches
    plt.title('Curve Optimization in a Custom Metric Space')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal') # Ensures x and y axes have the same scale
    
    plt.show()


if __name__ == "__main__":
    start = torch.tensor([1.0, -1.0])
    end = torch.tensor([1.0, 1.0])
    num_points = 5
    num_iterations = 100
    learning_rate = 0.01

    optimized_curve = train_curve(start, end, num_points, num_iterations, learning_rate)
    print("Optimized Curve:")
    print(optimized_curve)

    print("Generating plot...")
    plot_optimization_results(start, end, optimized_curve)