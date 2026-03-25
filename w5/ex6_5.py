import torch
from ex6_4 import calc_energy, plot_optimization_results, train_curve, parameterize_curve
#Generalized metric for batch size B
def metric(x_batch):
    # x_batch shape: (B, 2) where B is the number of midpoints
    B = x_batch.shape[0]
    
    p_x = torch.distributions.Normal(0.0, 0.1).log_prob(x_batch[:, 0]) # Shape: (B,)
    p_y = torch.distributions.Normal(0.0, 0.1).log_prob(x_batch[:, 1]) # Shape: (B,)

    # Combine the probabilities into a 2D tensor. Shape: (B, 2)
    p = torch.stack((p_x, p_y), dim=1)

    eps = 1e-6
    G = 1/(p + eps)

    # G now contains a 2x2 matrix for every single point in the batch!
    return G


def metric_from_pdf(x_batch, epsilon=1e-5):
    """
    Calculates a Riemannian metric based on the inverse of a 2D Standard Normal PDF.
    
    Args:
        x_batch (torch.Tensor): Shape (B, 2), the batch of midpoints.
        epsilon (float): A small constant to prevent division by zero.
        
    Returns:
        torch.Tensor: Shape (B, 2, 2), the batched metric matrices.
    """
    B = x_batch.shape[0]
    
    # 1. Calculate the squared norm of each point: ||x||^2
    sq_norm = torch.sum(x_batch**2, dim=1) # Shape: (B,)
    
    # 2. Calculate the 2D Standard Normal PDF for each point
    # p(x) = (1 / 2*pi) * exp(-0.5 * ||x||^2)
    coeff = 1.0 / (2.0 * math.pi)
    pdf_vals = coeff * torch.exp(-0.5 * sq_norm) # Shape: (B,)
    
    # 3. Convert the PDF into a cost (inverse probability)
    # High probability = low cost. Low probability = massive cost.
    cost = 1.0 / (pdf_vals + epsilon) # Shape: (B,)
    
    # 4. Create the batched 2x2 Identity matrices
    I = torch.eye(2, dtype=x_batch.dtype, device=x_batch.device).unsqueeze(0).expand(B, 2, 2)
    
    # 5. Multiply the scalar cost by the Identity matrix
    G = cost.view(B, 1, 1) * I
    
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
    G = metric_from_pdf(midpoints) 
    
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



if __name__ == "__main__":
    # Test the metric function with a batch of points
    test_points = torch.tensor([[0.0, 0.0]])
    G = metric(test_points)
    print("Metric G for test points:")
    print(G)

    start = torch.tensor([1.0, 0.0])
    end = torch.tensor([0.0, 1.0])
    num_points = 20
    num_iterations = 200
    learning_rate = 0.05

    # Generate the parameterized curve
    base_curve, interior = parameterize_curve(start, end, num_points)
    print("Base curve:")
    print(base_curve)   

    print("Training curve with custom metric...")
    optimized_curve = train_curve(start, end, num_points, num_iterations, learning_rate)

    print("\nFinal Curve Points:")
    print(optimized_curve)


    # Plot the optimized curve and its energy
    plot_optimization_results(start, end, optimized_curve, metric_fn=metric)


