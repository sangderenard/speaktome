import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from tqdm import tqdm
# ----- Parameters -----
volume_size = (5, 5, 5)  # Large 3D volume
batch_size = 5              # Number of parallel noise volumes
spectra_types = ['white', 'pink', 'brownian']  # Types of noise spectra
threshold = 0.75             # Detection confidence threshold
epochs = 50                  # Number of training epochs



# ----- Generate Noise Spectra -----
def generate_noise_spectrum(volume_size, batch_size, spectrum_types, training=True, device="cuda"):
    """
    Generate FFT-based noise for an entire batch with individually represented spectral profiles.
    """
    freq_grids = torch.meshgrid([
        torch.fft.fftshift(torch.linspace(-0.5, 0.5, size, device=device)) for size in volume_size
    ], indexing='ij')
    freq_magnitude = torch.sqrt(sum(f**2 for f in freq_grids))  # Shared grid

    batch_spectral_magnitudes = torch.zeros((batch_size, *freq_magnitude.shape))
    for i in tqdm(range(batch_size), desc="Generating Noise Spectra"):
        spectrum_type = spectrum_types[i % len(spectrum_types)]
        if spectrum_type == 'white':
            batch_spectral_magnitudes[i] = torch.ones_like(freq_magnitude)
        elif spectrum_type == 'pink':
            batch_spectral_magnitudes[i] = torch.where(freq_magnitude > 0, 1 / freq_magnitude, torch.tensor(0.0))
        elif spectrum_type == 'brownian':
            batch_spectral_magnitudes[i] = torch.where(freq_magnitude > 0, 1 / (freq_magnitude**2), torch.tensor(0.0))
        else:
            raise ValueError(f"Unknown spectrum type: {spectrum_type}")

    batch_phases = torch.rand((batch_size, *freq_magnitude.shape)) * 2 * np.pi
    batch_noise_fft = batch_spectral_magnitudes * torch.exp(1j * batch_phases)
    batch_noise = torch.real(fft.ifftn(batch_noise_fft, dim=(1, 2, 3)))

    return batch_noise  # Shape: [B, D, H, W]



# ----- Dynamic Ground Truth Generation -----
from itertools import combinations
from itertools import combinations
import torch
import torch.fft as fft
import numpy as np
from itertools import combinations
import torch
import torch.fft as fft
from tqdm import tqdm

def verify_frequency_sets(batch_volume, beat_threshold=1e-3, magnitude_threshold=1e-6, subbatch_size=50000000):
    """
    Detect voxel regions where three FFT frequencies form a valid dual-orbit beat pattern with full GPU parallelization.
    """
    B, D, H, W = batch_volume.shape
    device = batch_volume.device
    confidence_mask = torch.zeros((B, D, H, W), dtype=torch.float32, device=device)  # Full 3D binary mask

    for b in tqdm(range(B), desc="Verifying Frequency Sets"):  # Process each volume in the batch
        # Perform FFT and magnitude computation
        volume_fft = fft.fftn(batch_volume[b], dim=(0, 1, 2))
        magnitude = torch.abs(volume_fft)

        # Filter significant indices on GPU
        significant_mask = magnitude > magnitude_threshold
        significant_indices = torch.nonzero(significant_mask, as_tuple=False)
        if significant_indices.shape[0] < 3:
            continue

        magnitudes = magnitude[significant_mask]

        # Dynamically generate and process triplets
        triplet_generator = generate_triplets_gpu(significant_indices.shape[0], subbatch_size, device)
        for triplet_batch in triplet_generator:
            mark_voxels_parallel(confidence_mask[b], significant_indices, magnitudes, triplet_batch, beat_threshold, volume_fft.shape)

    return confidence_mask.unsqueeze(1)  # Shape: (B, 1, D, H, W)


def generate_triplets_gpu(num_indices, subbatch_size, device):
    """
    Dynamically generate triplets of indices on the GPU with tensor-based accumulation for efficiency.
    :param num_indices: Total number of significant indices
    :param subbatch_size: Maximum number of triplets per subbatch
    :param device: Torch device (e.g., 'cuda')
    :return: Generator yielding triplet subbatches
    """
    from tqdm import tqdm

    # Estimate total number of triplets
    total_triplets = (num_indices * (num_indices - 1) * (num_indices - 2)) // 6

    # Preallocate a tensor for accumulating triplets
    triplet_batch = torch.empty((subbatch_size, 3), dtype=torch.int32, device=device)
    batch_index = 0  # Tracks the current position in the batch

    # Initialize tqdm progress bar
    with tqdm(total=total_triplets, desc="Generating Triplets", leave=False) as pbar:
        # Use nested loops to generate triplets dynamically
        for i in range(num_indices):
            for j in range(i + 1, num_indices):
                for k in range(j + 1, num_indices):
                    # Insert triplet into the preallocated tensor
                    triplet_batch[batch_index] = torch.tensor([i, j, k], dtype=torch.int32, device=device)
                    batch_index += 1

                    # If the batch is full, yield it and reset
                    if batch_index >= subbatch_size:
                        yield triplet_batch.clone()  # Clone to avoid overwriting
                        pbar.update(batch_index)  # Update progress bar
                        batch_index = 0  # Reset index for the next batch

        # Yield any remaining triplets
        if batch_index > 0:
            yield triplet_batch[:batch_index].clone()  # Trim the batch to the actual size
            pbar.update(batch_index)



def mark_voxels_parallel(mask, significant_indices, magnitudes, triplets, beat_threshold, shape):
    """
    Vectorized evaluation of triplets and marking voxels on the GPU.
    """
    # Extract magnitudes for all triplets
    mag1 = magnitudes[triplets[:, 0]]
    mag2 = magnitudes[triplets[:, 1]]
    mag3 = magnitudes[triplets[:, 2]]

    # Vectorized beat frequency check
    freq_diff = torch.abs(mag1 - mag2)
    valid_triplets = torch.abs(freq_diff - mag3) < beat_threshold

    # Mark valid coordinates in the mask
    valid_indices = triplets[valid_triplets]
    coords = significant_indices[valid_indices.flatten()]
    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]
    mask[z, y, x] = 1.0


# ----- 3D Convolutional Kernel for Pattern Detection -----
class PatternDetector3D(nn.Module):
    def __init__(self):
        super(PatternDetector3D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Outputs confidence score between 0 and 1
        )

    def forward(self, x):
        return self.conv_layers(x)

# ----- Training Loop -----
def train_pattern_detector(model, epochs, batch_size, volume_size, device, spectrum_types):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0

        # Generate batch with progress bar
        batch_noise = generate_noise_spectrum(volume_size, batch_size, spectrum_types)
        batch_noise = batch_noise.unsqueeze(1).to(device)

        batch_ground_truth = verify_frequency_sets(batch_noise.squeeze(1)).to(device)

        outputs = model(batch_noise)
        loss = criterion(outputs, batch_ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


# ----- Continuous Batch Evaluation -----
def continuous_pattern_search(model, volume_size, batch_size, spectra_types, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    while True:
        batch_noise = []
        for _ in tqdm(range(batch_size), desc="Generating Noise Batches"):
            spectrum_type = np.random.choice(spectra_types)
            noise = generate_noise_spectrum(volume_size, 1, [spectrum_type])
            batch_noise.append(noise)
        batch_noise = torch.stack(batch_noise).to(device)

        with torch.no_grad():
            confidence_maps = model(batch_noise)

        for i, confidence_map in enumerate(confidence_maps):
            high_conf_indices = torch.nonzero(confidence_map > threshold, as_tuple=False)
            if len(high_conf_indices) > 0:
                tqdm.write(f"Batch {i}: Detected pattern at {high_conf_indices.cpu().numpy()}")

# ----- Run the Training and Continuous Search -----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatternDetector3D().to(device)

    print("Training the pattern detector...")
    train_pattern_detector(model, epochs, batch_size, volume_size, device, spectra_types)

    print("Running continuous pattern search...")
    continuous_pattern_search(model, volume_size, batch_size, spectra_types, threshold)
