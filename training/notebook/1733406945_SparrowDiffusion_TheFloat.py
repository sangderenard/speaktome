import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class TheFloat:
    def __init__(self, spatial_size):
        self.spatial_size = spatial_size
        self.field_metrics = {"mean": 0, "std": 0}
        self.gradient_field = np.zeros((spatial_size, spatial_size, 2))  # Gradient vector field

    def update(self, signal_field):
        """
        Update metrics and gradient field based on the signal field.
        """
        self.field_metrics["mean"] = np.mean(signal_field)
        self.field_metrics["std"] = np.std(signal_field)

        # Compute gradient
        grad_x, grad_y = np.gradient(signal_field)
        self.gradient_field = np.stack((grad_x, grad_y), axis=-1)

    def get_guidance_vector(self, position):
        """
        Get the vector guiding to the nearest signal region based on gradient field.
        """
        x, y = position.astype(int)
        x = np.clip(x, 0, self.spatial_size - 1)
        y = np.clip(y, 0, self.spatial_size - 1)
        return self.gradient_field[x, y]

class SparrowDiffusion:
    def __init__(self, num_doodads, spatial_size, brave_threshold, shy_threshold):
        self.num_doodads = num_doodads
        self.spatial_size = spatial_size
        self.positions = np.random.rand(num_doodads, 2) * spatial_size
        self.states = np.random.choice(["brave", "shy"], size=num_doodads)
        self.brave_threshold = brave_threshold
        self.shy_threshold = shy_threshold
        self.diffusion_rates = {"brave": 0.1, "shy": 0.02}

    def update_states(self, signal_field):
        """
        Update states based on local signal intensity.
        """
        for i, pos in enumerate(self.positions):
            x, y = pos.astype(int)
            x = np.clip(x, 0, signal_field.shape[0] - 1)
            y = np.clip(y, 0, signal_field.shape[1] - 1)
            signal_intensity = signal_field[x, y]

            if signal_intensity > self.brave_threshold:
                self.states[i] = "brave"
            elif signal_intensity < self.shy_threshold:
                self.states[i] = "shy"

    def diffuse(self, the_float):
        """
        Diffuse doodads based on their state and The Float's guidance.
        """
        for i, state in enumerate(self.states):
            rate = self.diffusion_rates[state]
            guidance = the_float.get_guidance_vector(self.positions[i])
            guidance = guidance / np.linalg.norm(guidance + 1e-8)  # Normalize to avoid NaNs
            self.positions[i] += guidance * rate
            # Keep positions within bounds
            self.positions[i] = np.clip(self.positions[i], 0, self.spatial_size - 1)

# Main Demonstration
if __name__ == "__main__":
    spatial_size = 100
    num_doodads = 200
    brave_threshold = 0.7
    shy_threshold = 0.3
    time_steps = 100

    # Create signal field (sinusoidal)
    x = np.linspace(0, 4 * np.pi, spatial_size)
    y = np.linspace(0, 4 * np.pi, spatial_size)
    X, Y = np.meshgrid(x, y)
    signal_field = 0.5 + 0.5 * np.sin(X) * np.cos(Y)  # Normalize to [0, 1]

    # Initialize The Float and SparrowDiffusion
    the_float = TheFloat(spatial_size)
    sparrow_diffusion = SparrowDiffusion(num_doodads, spatial_size, brave_threshold, shy_threshold)

    # Visualization setup
    fig, ax = plt.subplots()
    signal_img = ax.imshow(signal_field, cmap="viridis", origin="lower", extent=[0, spatial_size, 0, spatial_size])
    doodads_scatter = ax.scatter([], [], c=[], cmap="cool", alpha=0.7)

    def update(frame):
        # Update The Float and SparrowDiffusion
        the_float.update(signal_field)
        sparrow_diffusion.update_states(signal_field)
        sparrow_diffusion.diffuse(the_float)

        # Update scatter plot
        doodads_scatter.set_offsets(sparrow_diffusion.positions)
        colors = ["red" if state == "brave" else "blue" for state in sparrow_diffusion.states]
        doodads_scatter.set_array(np.array(colors))

        return signal_img, doodads_scatter

    ani = animation.FuncAnimation(fig, update, frames=time_steps, blit=False, repeat=False)
    plt.colorbar(signal_img, ax=ax, label="Signal Intensity")
    plt.title("The Float & SparrowDiffusion Demonstration")
    plt.show()
