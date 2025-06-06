import torch
import logging
import numpy as np
import os
import pickle
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import torchaudio
import time
import sys
# Import your YoungManAlgorithm class
from isosurface4 import YoungManAlgorithm
from triangulator import Triangulator
class TextureBufferGenerator:
    def __init__(self):
        self.texture_buffer_id = None

    def setup_texture(self, evaluation_result):
        """
        Create a texture from the evaluation result.

        Args:
            evaluation_result (dict): Output of the evaluation function, including vertices and indices.

        Returns:
            int: Texture buffer ID.
        """
        vertices = evaluation_result["vertices"].detach().cpu().numpy()
        indices = evaluation_result["indices"].detach().cpu().numpy()

        # Flatten and normalize vertex data for the texture
        vertex_data = vertices.flatten().astype(np.float32)

        # Create and bind the texture buffer
        if not self.texture_buffer_id:
            self.texture_buffer_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_BUFFER, self.texture_buffer_id)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, vertex_data)

        return self.texture_buffer_id

    def cleanup(self):
        """
        Cleanup the texture buffer.
        """
        if self.texture_buffer_id:
            glDeleteTextures([self.texture_buffer_id])

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Audio Configuration
CLEAR_MODES = ["full_clear", "decay_stacking", "full_blit"]


FFT_WINDOW_SIZE = 1024
FFT_OVERLAP = 0.5  # 50% overlap
FFT_REBINNING_FACTOR = 9  # Adjust for higher or lower frequency resolution
FREQUENCY_SCALE = .01
 
def scalar_sphere(x, y, z, r=1.0):
    """Example scalar function for a sphere."""
    return torch.sin(x)**2 + torch.sin(y)**2 + torch.sin(z)**2 - r**2

def initialize_pygame(width=800, height=600):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Live Isosurface Visualization')

def setup_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)
    
    # Set up projection matrix
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (800/600), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def setup_lighting():
    """Set up basic lighting in the OpenGL scene."""
    #glEnable(GL_LIGHTING)
    #glEnable(GL_LIGHT0)  # Enable Light 0

    # Set light position and properties
    light_position = [5.0, 5.0, 5.0, 1.0]  # x, y, z, w
    light_diffuse = [1.0, 1.0, 1.0, 1.0]  # RGB diffuse
    light_specular = [0.5, 0.5, 0.5, 1.0]  # RGB specular
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

    # Set material properties
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])  # High specular reflection
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)  # Shininess exponent


def handle_events(events, geometries, scalar_functions, processor, current_geometry, current_scalar_function, current_mode, num_modes, rotating, clear_mode):
    """
    Handle all events, including mode switching, geometry switching, and scalar function switching.

    Args:
        events (list): List of pygame events.
        geometries (list): Available geometries.
        scalar_functions (list): Available scalar functions.
        processor (YoungManAlgorithm): Geometry processor.
        current_geometry (str): Current geometry.
        current_scalar_function (int): Current scalar function index.
        current_mode (int): Current rendering mode.
        num_modes (int): Total number of rendering modes.

    Returns:
        tuple: Updated current_geometry, current_scalar_function, and current_mode.
    """
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_9:
                # Switch scalar function
                idx = event.key - pygame.K_1
                if idx < len(scalar_functions):
                    logging.info(f"Switching to Scalar Function {idx + 1}")
                    current_scalar_function = idx
            elif event.key == pygame.K_p:
                clear_mode = (clear_mode + 1) % len(CLEAR_MODES)
                logging.info(f"Switched to clearing mode: {CLEAR_MODES[clear_mode]}")

            elif pygame.K_a <= event.key <= pygame.K_z:
                # Switch geometry
                idx = event.key - pygame.K_a
                if idx < len(geometries):
                    geometry = geometries[idx]
                    logging.info(f"Switching to Geometry: {geometry}")
                    processor.switch_geometry(geometry)
                    current_geometry = geometry
                if event.key == pygame.K_r:
                    rotating = not rotating
            elif event.key == pygame.K_UP:
                # Increment rendering mode
                current_mode = (current_mode + 1) % num_modes
                logging.info(f"Switched to rendering mode {current_mode}")
            elif event.key == pygame.K_DOWN:
                # Decrement rendering mode
                current_mode = (current_mode - 1) % num_modes
                logging.info(f"Switched to rendering mode {current_mode}")
            elif event.key == pygame.K_ESCAPE:
                # Exit the program
                pygame.quit()
                quit()

    return current_geometry, current_scalar_function, current_mode, rotating, clear_mode
# Create and bind Vertex Buffer Object (VBO) for vertices
def setup_vbos(vertices, colors):
    vbo_ids = glGenBuffers(2)  # Generate two buffers: one for vertices, one for colors

    # Bind vertices to the first buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[0])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Bind colors to the second buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[1])
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)

    return vbo_ids
def generate_edge_texture(edges, vertices):
    # Create edge geometry using normalized vertices and edges
    edge_positions = vertices[edges.flatten()].reshape(-1, 3)
    # Use VBO for edge positions and return texture ID
    return setup_vbos(edge_positions, colors=None)

def render_with_vbos(vbo_ids=None, vertex_count=0, rotation_angle=0, mode=0, clear_mode=0, vao=None, vbo_positions=None, vbo_colors=None):
    """
    Render with VBOs or optionally with a VAO and position VBO.

    Args:
        vbo_ids (list): List of VBO IDs for vertices and colors (optional).
        vertex_count (int): Number of vertices to draw.
        rotation_angle (float): Rotation angle for rendering.
        mode (int): Rendering mode (0 for points, 1 for triangles).
        clear_mode (int): Clearing mode.
        vao (int): Vertex Array Object ID (optional).
        vbo_positions (int): Vertex Buffer Object ID for positions (optional).
    """
    if CLEAR_MODES[clear_mode] == "full_clear":
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Setup OpenGL for rendering
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5.0)
    glRotatef(rotation_angle, 1, 1, 0)

    if vao is not None and vbo_positions is not None:
        # Bind VAO if provided
        glBindVertexArray(vao)

        # Bind the position VBO if provided
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        # Bind the color VBO if provided
        if vbo_colors:
            
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(4, GL_FLOAT, 0, None)

        # Draw the VBO
        glDrawArrays(GL_POINTS if mode == 0 else GL_TRIANGLES, 0, vertex_count)

        # Unbind VAO
        glBindVertexArray(0)
        glDisableClientState(GL_VERTEX_ARRAY)
    elif vbo_ids is not None:
        # Enable VBO client states
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[0])
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, vbo_ids[1])
        glColorPointer(4, GL_FLOAT, 0, None)

        # Draw the VBO
        glDrawArrays(GL_POINTS if mode == 0 else GL_TRIANGLES, 0, vertex_count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
    else:
        raise ValueError("Either `vbo_ids` or `vao` and `vbo_positions` must be provided.")

    # Create a texture from the framebuffer
    texture_width, texture_height = pygame.display.get_surface().get_size()
    rendered_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, rendered_texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, texture_width, texture_height, 0)

    # Return the rendered texture
    return rendered_texture


#OVERLAY_MIX = 0.5  # Set the mix ratio (0 = full overlay, 1 = full split-screen)

def blend_textures_with_center_offset(texture_a, texture_b, overlay_mix=0.5):
    """
    Blend two textures additively with center offsets.
    """
    # Texture dimensions
    texture_width, texture_height = pygame.display.get_surface().get_size()

    # Create framebuffer
    framebuffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

    # Blended texture
    blended_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, blended_texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_width, texture_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blended_texture, 0)

    # Clear framebuffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)

    # Disable VAO/VBO to avoid conflicts
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Additive blending
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    # Render texture_a
    glBindTexture(GL_TEXTURE_2D, texture_a)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(-1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, 1)
    glTexCoord2f(0, 1); glVertex2f(-1, 1)
    glEnd()

    # Render texture_b
    glBindTexture(GL_TEXTURE_2D, texture_b)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(-1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, 1)
    glTexCoord2f(0, 1); glVertex2f(-1, 1)
    glEnd()

    # Reset state
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDisable(GL_BLEND)

    return blended_texture




def render_opengl_data(evaluation_result, rotation_angle, mode, triangulator=None, rotating=False, clear_mode=0):
    # Process evaluation result and generate VBOs
    vertices = evaluation_result["vertices"].detach().cpu().numpy()
    indices = evaluation_result["indices"].detach().cpu().numpy()

    # Normalize vertices
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    x_normalized = (x + np.pi) / (2 * np.pi)
    y_normalized = (y + np.pi) / (2 * np.pi)
    z_normalized = (z + np.pi) / (2 * np.pi)

    brightness = np.full(vertices.shape[0], 2.0)
    colors = np.stack([x_normalized, y_normalized, z_normalized], axis=1) * brightness[:, np.newaxis]
    alpha = np.full(vertices.shape[0], 1.0)
    colors_with_alpha = np.clip(np.concatenate([colors, alpha[:, np.newaxis]], axis=1), 0.0, 1.0)

    # Generate VBOs
    vbo_ids = setup_vbos(vertices.astype(np.float32), colors_with_alpha.astype(np.float32))

    # Render to texture
    texture = render_with_vbos(vbo_ids, vertices.shape[0], rotation_angle, mode, clear_mode)
    # Use line diagram buffers for additional rendering
    #if "line_diagram" in evaluation_result["passive_data"]["evaluation_dataset"]:
        #vao, vbo_positions, vbo_colors, num_vertices = evaluation_result["passive_data"]["evaluation_dataset"]["line_diagram"]
        
        #alsotexture = render_with_vbos(vao=vao, vbo_positions=vbo_positions, vertex_count=num_vertices, vbo_colors=vbo_colors)
        # Bind the VAO and VBO for positions
        #glBindVertexArray(vao)
        #glBindBuffer(GL_ARRAY_BUFFER, vbo_positions)
        #glDrawArrays(GL_POINTS, 0, num_vertices)  # Draw lines using the vertices in the buffer



        # Similarly, inspect the color buffer
        #glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
        #color_data = np.empty(len(vertices) * 2 * 4, dtype=np.float16)  # 4 components (RGBA)
        #glGetBufferSubData(GL_ARRAY_BUFFER, 0, color_data.nbytes, color_data)

        # Reshape and log the color data
        #color_data = color_data.reshape(-1, 4)  # Each vertex has 4 color components
        #print("Line Buffer Colors:")
        #print(color_data)

        #glBindVertexArray(0)

    # Placeholder: Generate edge-point texture (example logic, replace as needed)
    #edge_point_texture = glGenTextures(1)
    #glBindTexture(GL_TEXTURE_2D, edge_point_texture)
    #glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    # Blend the textures
    #blended_texture = blend_textures_with_center_offset(texture, alsotexture)

    # Bind and render the final blended texture
    glBindTexture(GL_TEXTURE_2D, texture)
    #glBindTexture(GL_TEXTURE_2D, blended_texture)
    #glBegin(GL_QUADS)
    #glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
    #glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0)
    #glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
    #glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0)
    #glEnd()
    pygame.display.flip()
from queue import Queue
import sounddevice as sd

# Queue for live audio data
audio_buffer = Queue()

def audio_callback(indata, frames, time, status):
    """
    Callback to capture live audio into a buffer.
    """
    if status:
        logging.warning(f"Audio stream status: {status}")
    audio_buffer.put(indata[:, 0])  # Push mono audio into the buffer

def start_audio_stream(sample_rate=44100, channels=1):
    """
    Start a live audio stream and feed data into the queue.
    """
    stream = sd.InputStream(callback=audio_callback, samplerate=sample_rate, channels=channels)
    stream.start()
    logging.info("Live audio stream started.")
    return stream


def main():
    # Path to your audio file
    audio_file_path = r'Afraid No More.wav'  # Replace with your actual audio file path

    # Load audio data
    waveform, sample_rate = torchaudio.load(audio_file_path)

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    device = "cuda"
    # Move waveform to CPU
    waveform = waveform.to('cpu')

    total_samples = waveform.shape[-1]

    # Initialize Pygame mixer and play audio
    pygame.mixer.init(frequency=sample_rate)
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()

    # Record start time
    start_time = time.time()

    # Initialize Pygame and OpenGL
    initialize_pygame()
    setup_opengl()
    setup_lighting()

    # Available geometries and scalar functions
    geometries = ["tetrahedron", "cube", "icosahedron", "octahedron", "square"]
    scalar_functions = [
        lambda x, y, z, t: x + y + z,
        lambda x, y, z, t: x**2 + y**2 + z**2 - 1.0,  # Sphere
        lambda x, y, z, t: torch.sin(x + y + z + t),  # Sine wave propagation
        lambda x, y, z, t: torch.cos(x * y * z + t),  # Oscillating cosine wave
        lambda x, y, z, t: x**2 - y**2 + z**2 - 0.5,  # Ellipsoid-like function
        lambda x, y, z, t: torch.sqrt(x**2 + y**2 + z**2) - 1.0,  # Shell
        lambda x, y, z, t: torch.atan(x**2 + y**2 + z**2) - 1.0,  # Shell
        lambda x, y, z, t: torch.sin(x + t) * torch.cos(y + t) * torch.sin(z + t)  # Ripple sphere
    ]

    # Initialize YoungManAlgorithm and Triangulator
    density = 5
    domain_bounds = [
        (-torch.pi*2/3, torch.pi*2/3),  # x range
        (-torch.pi*2/3, torch.pi*2/3),  # y range
        (-torch.pi*2/3, torch.pi*2/3),  # z range
    ]
    isovalue = 0.0
    
    processor = YoungManAlgorithm(geometry=geometries[0], density=density, jitter_enabled=False, micro_jitter=False, precision=torch.float32, device=device)
    triangulator = Triangulator()
    rotating = False
    clear_mode = 0  # Start with full_clear
    # Default geometry, scalar function, and rendering mode
    current_geometry = geometries[0]
    current_scalar_function = 0
    current_mode = 0

    rotation_angle = 0
    clock = pygame.time.Clock()
    current_time = torch.tensor([0.0], device=device)

    running = True
    while running:
        # Handle events for mode switching
        events = pygame.event.get()

        current_geometry, current_scalar_function, current_mode, rotating, clear_mode = handle_events(
            events, geometries, scalar_functions, processor, current_geometry, current_scalar_function, current_mode, 5, rotating, clear_mode
        )

        # Get current playback time
        current_time += .01
        elapsed_time = current_time - start_time
        current_sample_index = int(elapsed_time * sample_rate) % total_samples

        # Extract audio window for FFT
        window_size = FFT_WINDOW_SIZE
        step_size = int(window_size * (1 - FFT_OVERLAP))
        start_index = current_sample_index - window_size + 1
        if start_index < 0:
            # Wrap around
            audio_window = torch.cat((waveform[..., start_index:], waveform[..., :current_sample_index + 1]), dim=-1)
        else:
            audio_window = waveform[..., start_index:current_sample_index + 1]

        # Ensure audio_window has the correct size
        if audio_window.shape[-1] < window_size:
            # Not enough data, skip this frame
            continue
        else:
            audio_window = audio_window[..., -window_size:]

        # Compute FFT using PyTorch
        windowed_data = audio_window * torch.hann_window(window_size)
        fft_result = torch.fft.rfft(windowed_data)
        frequencies = torch.fft.rfftfreq(window_size, d=1.0 / sample_rate)

        # Rebinning FFT data if necessary
        if FFT_REBINNING_FACTOR > 1:
            fft_result = fft_result.reshape(-1, FFT_REBINNING_FACTOR).mean(dim=1)
            frequencies = frequencies.reshape(-1, FFT_REBINNING_FACTOR).mean(dim=1)

        fft_magnitudes = torch.abs(fft_result)
        fft_phases = torch.angle(fft_result)

        # Normalize FFT magnitudes
        fft_magnitudes /= fft_magnitudes.max() + 1e-8

        # Move data to GPU if necessary
        fft_magnitudes = fft_magnitudes.to(device)
        fft_phases = fft_phases.to(device)
        frequencies = frequencies.to(device)

        # Use frequencies and phases to perturb the scalar function
        t = torch.tensor([elapsed_time], device=device) * FREQUENCY_SCALE

        # Get the active scalar function
        base_scalar_function = scalar_functions[current_scalar_function]
        def dynamic_scalar_function(x, y, z):
            """
            Apply FFT-based perturbations to a scalar field, spatially and temporally dependent.

            Args:
                x, y, z (torch.Tensor): Spatial coordinates of the domain.

            Returns:
                torch.Tensor: Perturbed scalar field.
            """

            # Base scalar field at all points
            base_value = base_scalar_function(x, y, z, t=current_time)  # Shape: (num_points,)
            
            # Compute spatial distance (norm) for wave propagation
            spatial_norm = torch.sqrt(x**2 + y**2 + z**2)  # Shape: (num_points,)

            # Broadcast FFT components for computation
            frequencies_expanded = frequencies.view(1, -1)  # Shape: (1, num_freqs)
            phases_expanded = fft_phases.view(1, -1)        # Shape: (1, num_freqs)
            magnitudes_expanded = fft_magnitudes.view(1, -1)  # Shape: (1, num_freqs)

            # Compute sinusoidal perturbation at all spatial points
            # The spatial norm interacts with FFT frequencies to create unique perturbations per point
            perturbation = magnitudes_expanded * torch.sin(
                2 * torch.pi * frequencies_expanded * spatial_norm.unsqueeze(-1) + phases_expanded
            )  # Shape: (num_points, num_freqs)

            # Sum perturbations over all frequencies
            total_perturbation = perturbation.sum(dim=-1)  # Shape: (num_points,)

            # Scale perturbation for smoothness
            scaling_factor = 1.1
            total_perturbation *= scaling_factor

            # Add perturbation to the base scalar field
            return base_value + total_perturbation


        # Evaluate the scalar field for the current geometry
        evaluation_result = processor.evaluate(
            dynamic_scalar_function,
            domain_bounds,
            isovalue=isovalue,
            gradient_normals=False,
            compute_vertex_normals=False,
            centroid_refinement=False,
            deduplicate=False,
            oversampling_grid=(1, 1, 1),
            oversampling_spot=1,
            jitter_strength=(0.0,0.0,.0)
        )

        # Render the updated data using the current mode
        render_opengl_data(evaluation_result, rotation_angle, current_mode, triangulator, rotating, clear_mode)

        # Update rotation angle
        if rotating:
            rotation_angle += 1
            if rotation_angle >= 360:
                rotation_angle -= 360

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
