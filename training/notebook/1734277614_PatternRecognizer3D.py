import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

# ----- Parameters -----
volume_size = (32, 32, 32)  # 3D grid size
energy_threshold = 1e-3     # Cooling threshold
energy_introduction = 0.1   # Initial energy envelope

# ----- Generate Spectrally Defined Noise -----
def generate_spectral_noise(volume_size, spectral_profile):
    """
    Generate noise in the Fourier domain with a defined spectral profile.
    - spectral_profile: A callable function defining spectral magnitude based on frequency.
    """
    freq_grid = torch.meshgrid([
        torch.fft.fftshift(torch.linspace(-0.5, 0.5, size)) for size in volume_size
    ], indexing='ij')
    freq_magnitude = torch.sqrt(sum(f**2 for f in freq_grid))

    # Apply spectral profile (e.g., 1/f noise, Gaussian decay, etc.)
    spectral_magnitude = spectral_profile(freq_magnitude)

    # Random phases in Fourier domain
    phases = torch.rand_like(spectral_magnitude) * 2 * np.pi
    noise_fft = spectral_magnitude * torch.exp(1j * phases)

    # Inverse FFT to return to spatial domain
    noise = torch.real(fft.ifftn(noise_fft))
    return noise

# Example Spectral Profile: 1/f Noise
def spectral_profile_1_over_f(freq_magnitude):
    return torch.where(freq_magnitude > 0, 1 / freq_magnitude, torch.tensor(0.0))

# ----- Define 3D CNN for Pattern Recognition -----
class PatternRecognizer3D(nn.Module):
    def __init__(self):
        super(PatternRecognizer3D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Outputs confidence between 0 and 1
        )

    def forward(self, x):
        return self.conv_layers(x)

# ----- Initialize Volume and CNN -----
volume = generate_spectral_noise(volume_size, spectral_profile_1_over_f)
cnn = PatternRecognizer3D()

# ----- Main Loop for Edge Induction -----
def induce_edges(volume, cnn, energy_threshold):
    energy = energy_introduction
    vertices = []
    edges = []

    while energy > energy_threshold:
        # Step 1: Compute FFT and magnitude
        fft_data = fft.fftn(volume)
        magnitude = torch.abs(fft_data)
        
        # Step 2: Pass FFT data to 3D CNN
        cnn_input = magnitude.unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
        confidence_map = cnn(cnn_input).squeeze()        # Output confidence map

        # Step 3: Identify high-confidence regions
        high_conf_indices = torch.nonzero(confidence_map > 0.9, as_tuple=False)
        for idx in high_conf_indices:
            # Create vertices and edges based on pattern position
            vertex = tuple(idx.tolist())
            if vertex not in vertices:
                vertices.append(vertex)
            for v in vertices:  # Form edges to nearby vertices
                if vertex != v and torch.dist(torch.tensor(vertex), torch.tensor(v)) < 5:
                    edges.append((vertex, v))

        # Step 4: Subtract energy at confident regions
        for idx in high_conf_indices:
            volume[idx[0], idx[1], idx[2]] *= 0.5  # Damp energy locally

        # Step 5: Update energy & check cooling condition
        energy = volume.abs().mean().item()
        print(f"Energy: {energy:.6f}, Vertices: {len(vertices)}, Edges: {len(edges)}")

    return vertices, edges

# ----- Run the Simulation -----
vertices, edges = induce_edges(volume, cnn, energy_threshold)
print("Simulation complete.")
print(f"Final Vertices: {len(vertices)}, Final Edges: {len(edges)}")
