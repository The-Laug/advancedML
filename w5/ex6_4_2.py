import torch
import torch.nn as nn
from ex6_4 import metric, calc_energy, plot_optimization_results


class TextbookPolyCurve(nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
        
        # w1 corresponds to t, w2 corresponds to t^2
        self.w1 = nn.Parameter(torch.zeros(2, dtype=start.dtype, device=start.device))
        self.w2 = nn.Parameter(torch.zeros(2, dtype=start.dtype, device=start.device))
        
    def forward(self, num_points):
        t = torch.linspace(0, 1, num_points, device=self.start.device).unsqueeze(1)
        
        # 1. The Linear Base: (1 - t)c0 + tc1
        linear_part = (1 - t) * self.start + t * self.end
        
        # 2. Enforce the boundary constraint for the nonlinear remainder: w3 = -(w1 + w2)
        w3 = -(self.w1 + self.w2)
        
        # 3. The Nonlinear Remainder: w3*t^3 + w2*t^2 + w1*t
        nonlinear_part = w3 * t**3 + self.w2 * t**2 + self.w1 * t
        
        # Add them together!
        return linear_part + nonlinear_part


def train_poly_curve(start, end, num_points, num_iterations, learning_rate):
    # Use our new Textbook formulation
    poly_model = TextbookPolyCurve(start, end)
    optimizer = torch.optim.Adam(poly_model.parameters(), lr=learning_rate)

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        curve = poly_model(num_points)
        energy = calc_energy(curve) # Using your original calc_energy
        energy.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print(f"Iteration {iteration}, Energy: {energy.item():.4f}")

    return poly_model(num_points).detach(), poly_model

if __name__ == "__main__":
    # CHANGED: Start and end points that don't pass through the origin
    start = torch.tensor([1.0, 0.0])
    end = torch.tensor([0.0, 1.0])
    num_points = 20
    num_iterations = 200
    learning_rate = 0.05 

    print("Training Textbook Polynomial Curve...")
    optimized_poly_curve, final_model = train_poly_curve(start, end, num_points, num_iterations, learning_rate)
    
    print("\nFinal Polynomial Parameters:")
    for name, param in final_model.named_parameters():
        print(f"{name}: {param.detach().numpy()}")

    # Plot the optimized curve and its energy
    plot_optimization_results(start, end, optimized_poly_curve)
    