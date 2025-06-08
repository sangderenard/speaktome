# test_pattern_generator.py

import torch

class TestPatternGenerator:
    def __init__(self, device):
        self.device = device
        self.pattern_cache = {}  # Cache patterns by size (N)

    def generate_noise_texture(self, size, noise_type='pink'):
        h, w = size
        # Generate white noise
        white_noise = torch.randn((h, w), device=self.device)

        # Perform FFT
        f = torch.fft.fft2(white_noise)
        fshift = torch.fft.fftshift(f)

        # Create frequency grid
        y = torch.linspace(-0.5, 0.5, h, device=self.device).unsqueeze(1).repeat(1, w)
        x = torch.linspace(-0.5, 0.5, w, device=self.device).unsqueeze(0).repeat(h, 1)
        radius = torch.sqrt(x**2 + y**2)
        radius[radius == 0] = 1e-6  # Avoid division by zero

        # Apply filter for pink or brown noise
        if noise_type == 'pink':
            filter = 1 / radius
        elif noise_type == 'brown':
            filter = 1 / (radius ** 2)
        else:
            filter = torch.ones_like(radius)

        f_filtered = fshift * filter

        # Inverse FFT to get noise texture
        f_ishift = torch.fft.ifftshift(f_filtered)
        noise_texture = torch.fft.ifft2(f_ishift).real

        # Normalize to 0-255 and convert to uint8
        noise_texture = noise_texture - noise_texture.min()
        noise_texture = noise_texture / noise_texture.max()
        noise_texture = (noise_texture * 255).clamp(0, 255).to(torch.uint8)

        return noise_texture

    def create_checkerboard(self, size, tile_size):
        h, w = size
        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing='ij'
        )
        checkerboard = ((yy // tile_size) + (xx // tile_size)) % 2
        return checkerboard

    def get_test_pattern(self, N, n_slices):
        """
        Returns the test pattern for the given data length N and number of slices.
        Caches the pattern based on N.
        """
        if N in self.pattern_cache:
            return self.pattern_cache[N]

        # Calculate dimensions of the texture
        h = int(N ** 0.5)
        w = h
        if h * w < N:
            w += 1  # Adjust width if necessary

        size = (h, w)

        # Generate pink and brown noise textures
        pink_noise = self.generate_noise_texture(size, noise_type='pink')
        brown_noise = self.generate_noise_texture(size, noise_type='brown')

        # Create checkerboard mask
        tile_size = max(h // 8, 1)  # Ensure tile_size is at least 1
        checkerboard = self.create_checkerboard(size, tile_size)

        # Combine noises using the checkerboard mask
        test_pattern = torch.where(checkerboard == 0, pink_noise, brown_noise)

        # Flatten the test pattern and match the number of vertices
        test_pattern = test_pattern.flatten()[:N]

        # Repeat the pattern to match the n_slices
        test_pattern = test_pattern.unsqueeze(0).repeat(n_slices, 1)

        # Convert to RGB by stacking the same pattern
        combined_phase_buffer = torch.stack([test_pattern, test_pattern, test_pattern], dim=-1)

        # Cache the pattern
        self.pattern_cache[N] = combined_phase_buffer

        return combined_phase_buffer
