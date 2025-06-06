import torch
from laplacehelper import LaplaceHelper
from laplace import GridDomain, BuildLaplace3D, Transform
import matplotlib.pyplot as plt

# Time-Dependent Simulation Demonstration
def time_dependent_simulation_demo():
    print("\n--- Starting Time-Dependent Laplace Simulations ---\n")

    # Initialize the Laplace helper
    laplace = LaplaceHelper()
    laplace.setup_fields()

    # Add a custom metric modifier for scaling over time
    laplace.add_metric_modifier(
        "time_scaling",
        lambda tensor, t: tensor + torch.eye(3) * t
    )

    # Initialize 3D grid
    resolution = 30
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # Grid extents
    U, V, W = torch.meshgrid(
        torch.linspace(0, Lx, resolution),
        torch.linspace(0, Ly, resolution),
        torch.linspace(0, Lz, resolution),
        indexing="ij"
    )

    # Define grid domain
    transform = Transform(uextent=Lx, vextent=Ly, grid_boundaries=(True, True, True, True, True, True))
    transform.transform_spatial = lambda U, V, W: (U, V, W)
    grid_domain = GridDomain(U, V, W, transform=transform)

    # Initialize the 3D Laplace solver
    laplace_solver = BuildLaplace3D(
        grid_domain=grid_domain,
        metric_tensor_func=laplace.metric_modifier.apply_modifiers
    )

    # Simulation parameters
    time_steps = 5
    time_interval = 0.2  # Increment time in each step
    field_evaluations = []

    # Time-stepped simulation loop
    for t in range(time_steps):
        print(f"\n--- Time Step {t+1}: t = {t * time_interval} ---")

        # Modify metric tensor dynamically based on time
        laplace.modify_metric(t * time_interval)

        # Evaluate fields at a specific location (center)
        center_x, center_y, center_z = 0.5, 0.5, 0.5
        field_value = laplace.evaluate_fields(center_x, center_y, center_z)
        field_evaluations.append(field_value)

        # Compute Laplace solution
        laplace_tensor, _ = laplace_solver.build_general_laplace(U, V, W, f=0.0)
        print(f"Laplace Tensor Shape at t = {t * time_interval}: {laplace_tensor.shape}")

    # Plot field evaluations over time
    plt.plot([i * time_interval for i in range(time_steps)], field_evaluations, marker='o')
    plt.title("Time-Dependent Field Evaluation")
    plt.xlabel("Time (t)")
    plt.ylabel("Field Value at Center")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    time_dependent_simulation_demo()
