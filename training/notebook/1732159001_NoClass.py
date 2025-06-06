import json
import numpy as np
import matplotlib.pyplot as plt

def gradient_json_to_heatmap(json_file, focus_filter=None):
    """
    Visualizes gradient norms from a JSON file as a heatmap.
    
    Args:
        json_file (str): Path to the JSON file containing gradient history.
        focus_filter (str, optional): Filter by "generator" or "discriminator".
                                      If None, include all entries.
    """
    # Load JSON data
    with open(json_file, "r") as f:
        gradient_data = json.load(f)
    
    # Filter by focus if specified
    if focus_filter:
        gradient_data = [entry for entry in gradient_data if entry["focus"] == focus_filter]

    # Extract timestamps and gradient norms
    timestamps = [entry["timestamp"] for entry in gradient_data]
    grad_norms = [entry["grad_norm"] for entry in gradient_data]

    # Normalize timestamps to relative time
    min_time = min(timestamps)
    timestamps = [t - min_time for t in timestamps]

    # Prepare heatmap data
    # Convert to 2D array by grouping norms into bins based on relative time
    num_bins = 100  # Number of time bins
    time_bins = np.linspace(0, max(timestamps), num_bins)
    binned_grad_norms = [[] for _ in range(num_bins)]

    for t, norm in zip(timestamps, grad_norms):
        bin_index = np.digitize(t, time_bins) - 1
        binned_grad_norms[bin_index].append(norm)

    # Take the mean of each bin for the heatmap
    mean_grad_norms = [np.mean(bin) if bin else 0 for bin in binned_grad_norms]

    # Reshape for heatmap (assuming a grid with one row per parameter or layer)
    heatmap_data = np.array(mean_grad_norms).reshape(1, -1)

    # Plot heatmap
    plt.figure(figsize=(12, 2))
    plt.imshow(heatmap_data, aspect="auto", cmap="viridis", extent=[0, max(timestamps), 0, 1])
    plt.colorbar(label="Gradient Norm")
    plt.title(f"Gradient Heatmap ({focus_filter or 'All'})")
    plt.xlabel("Time (relative)")
    plt.ylabel("Layers/Parameters")
    plt.tight_layout()
    plt.show()

# Example usage:
gradient_json_to_heatmap("gradient_history.json", focus_filter="generator")
