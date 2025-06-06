import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_demo_data(num_bitmasks, num_vertices, dim=3):
    """
    Generate synthetic data to validate the encode_bitmask_characteristics function.
    """
    archetypal_offsets = torch.randn(num_vertices, dim)
    bitmask_to_activation_map = torch.randint(0, 2, (num_bitmasks, num_vertices))
    activation_centroids = torch.randn(num_bitmasks, dim)
    return archetypal_offsets, bitmask_to_activation_map, activation_centroids

def encode_bitmask_characteristics(
    archetypal_offsets, 
    bitmask_to_activation_map, 
    activation_centroids, 
    assumption_set="planar",
    precision=torch.float32
):
    """
    Encodes characteristics of intersection patterns into a binary integer matrix.
    """
    num_bitmasks, num_vertices = bitmask_to_activation_map.shape
    dim = archetypal_offsets.size(-1)

    characteristics = torch.zeros((num_bitmasks, 8), dtype=precision, device=archetypal_offsets.device)

    active_offsets = bitmask_to_activation_map.unsqueeze(-1) * archetypal_offsets.unsqueeze(0)
    relative_offsets = active_offsets - activation_centroids.unsqueeze(1)

    if assumption_set == "planar":
        plane_normals = torch.cross(relative_offsets[:, 1, :] - relative_offsets[:, 0, :], 
                                    relative_offsets[:, 2, :] - relative_offsets[:, 0, :], dim=-1)
        plane_normals = plane_normals / (plane_normals.norm(dim=-1, keepdim=True) + 1e-8)
        normal_deviation = torch.abs(torch.matmul(relative_offsets, plane_normals.unsqueeze(-1)).squeeze(-1))
        characteristics[:, 0] = normal_deviation.mean(dim=1)
        characteristics[:, 1] = normal_deviation.std(dim=1)
    elif assumption_set == "spherical":
        radii = relative_offsets.norm(dim=-1)
        mean_radius = radii.mean(dim=1)
        characteristics[:, 0] = (radii - mean_radius.unsqueeze(-1)).abs().mean(dim=1)
        characteristics[:, 1] = (radii - mean_radius.unsqueeze(-1)).abs().std(dim=1)
    else:
        raise ValueError(f"Unsupported assumption set: {assumption_set}")

    characteristics[:, 2] = bitmask_to_activation_map.sum(dim=1)
    characteristics[:, 3] = relative_offsets.norm(dim=-1).max(dim=1).values

    normalized = (characteristics - characteristics.min(dim=0).values) / (
        characteristics.max(dim=0).values - characteristics.min(dim=0).values + 1e-8
    )
    binary_encoded = (normalized * 255).to(torch.int32)

    return binary_encoded

def plot_loss_vs_time(confidence_map, activation_centroids):
    """
    Illustrates the loss compared to the time domain used for analysis.
    """
    time_steps = np.linspace(0, 10, confidence_map.size(0))
    loss = torch.norm(activation_centroids - activation_centroids.mean(dim=0), dim=-1).numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, loss, label="Loss")
    plt.xlabel("Time")
    plt.ylabel("Loss (Deviation from Mean)")
    plt.title("Loss vs Time Domain")
    plt.legend()
    plt.grid()
    plt.show()

def demonstrate_characteristics_analysis():
    """
    Demonstrate and validate the encode_bitmask_characteristics function using synthetic data.
    """
    num_bitmasks = 256
    num_vertices = 8
    dim = 3

    archetypal_offsets, bitmask_to_activation_map, activation_centroids = generate_demo_data(num_bitmasks, num_vertices, dim)

    confidence_map = encode_bitmask_characteristics(
        archetypal_offsets, 
        bitmask_to_activation_map, 
        activation_centroids, 
        "planar"
    )

    print("Confidence Map Shape:", confidence_map.shape)
    print("Sample Confidence Map:", confidence_map[:5])

    plot_loss_vs_time(confidence_map, activation_centroids)

# Run the demonstration
demonstrate_characteristics_analysis()
