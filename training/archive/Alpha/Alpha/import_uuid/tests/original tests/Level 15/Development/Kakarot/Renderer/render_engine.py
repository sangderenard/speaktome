# renderer/render_engine.py

import numpy as np
import torch
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from .lighting import Lighting
from .counter_translator import CounterTranslator
import pygame
import os

class RenderEngine:
    def __init__(self, config):
        self.config = config
        self.init_shaders()
        self.projection_matrix = None
        self.view_matrix = None
        self.lighting = Lighting()
# Initialize buffers during setup
        # Set up a Transform Feedback buffer for capturing outputAperture
        self.current_aperture = 1
        self.dof_color_texture = None
        self.dof_distance_texture = None
        self.dof_normal_texture = None
        self.blur_texture = None
        self.interest_score_texture = None
        self.focal_clarity_texture = None
        self.old_texture = None
        self.deep_buffer_defs = {
            0: {'attachment': GL_COLOR_ATTACHMENT0, 'texture' : self.dof_color_texture, 'name': 'dof_color_texture'},
            1: {'attachment': GL_COLOR_ATTACHMENT1, 'texture' : self.dof_distance_texture, 'name': 'dof_distance_texture'},
            2: {'attachment': GL_COLOR_ATTACHMENT2, 'texture' : self.dof_normal_texture, 'name': 'dof_normal_texture'},
        }
        self.shallow_buffer_defs = {
            0: {'attachment': GL_COLOR_ATTACHMENT0, 'texture' : self.interest_score_texture, 'name': 'interest_score_texture'},
            1: {'attachment': GL_COLOR_ATTACHMENT1, 'texture' : self.focal_clarity_texture, 'name': 'focal_clarity_texture'},
            2: {'attachment': GL_COLOR_ATTACHMENT2, 'texture' : self.old_texture, 'name': 'old_texture'},
            3: {'attachment': GL_COLOR_ATTACHMENT3, 'texture' : self.blur_texture, 'name': 'blur_texture'},
        }
        self.buffers = [ self.deep_buffer_defs, self.shallow_buffer_defs ]
        self.deep_fbo, self.shallow_fbo = self.initialize_buffers()          
        self.main_fbos = [self.deep_fbo, self.shallow_fbo]
        self.selected_output = (1,3)  # Default selected output location for display


    def draw_ground_plane(self):
        glBegin(GL_QUADS)
        glColor3f(0.4, 0.8, 0.4)
        size = 100
        glVertex3f(-size, 0.0, -size)
        glVertex3f(size, 0.0, -size)
        glVertex3f(size, 0.0, size)
        glVertex3f(-size, 0.0, size)
        glEnd()
        

    def set_selected_output(self, location):
        self.selected_output = location

    def activate_all_draw_buffers(self, fbo, buffers):
        draw_buffers = [buffer_info['attachment'] for buffer_info in buffers.values()]
        glDrawBuffers(len(draw_buffers), (GLuint * len(draw_buffers))(*draw_buffers))


    def activate_selected_draw_buffer(self):
        """Activates only the selected buffer for single texture rendering."""
        selected_attachment = [self.buffers[self.selected_output[0]][self.selected_output[1]]['attachment']]
        glDrawBuffers(len(selected_attachment), selected_attachment)

    def compile_and_check_shader(self, shader_code, shader_type):
        """Compile a shader and check for compilation errors."""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, shader_code)
        glCompileShader(shader)
        
        # Check if shader compiled successfully
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            error = glGetShaderInfoLog(shader).decode('utf-8')
            shader_name = "Vertex" if shader_type == GL_VERTEX_SHADER else (
                          "Geometry" if shader_type == GL_GEOMETRY_SHADER else "Fragment")
            print(f"Error compiling {shader_name} Shader:\n{error}")
            raise RuntimeError(f"{shader_name} shader compilation failed.")
        
        return shader

    def link_and_check_program(self, shaders):
        """Link shaders into a program and check for linking errors."""
        program = glCreateProgram()
        for shader in shaders:
            glAttachShader(program, shader)

        glLinkProgram(program)

        # Check if program linked successfully
        if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
            error = glGetProgramInfoLog(program).decode('utf-8')
            print(f"Error linking shader program:\n{error}")
            raise RuntimeError("Shader program linking failed.")

        # Detach and delete shaders after linking
        for shader in shaders:
            glDetachShader(program, shader)
            glDeleteShader(shader)

        return program
    def init_shaders(self):
        def load_shader_source(file_path):
            with open(file_path, 'r') as file:
                return file.read()

        # Load and compile the shaders for each stage of the rendering pipeline
        vertex_shader_code = load_shader_source(os.path.join(os.path.dirname(__file__), "shaders\\phenom.glsl"))
        geometry_shader_code = load_shader_source(os.path.join(os.path.dirname(__file__), "shaders\\form.glsl"))
        fragment_shader_code = load_shader_source(os.path.join(os.path.dirname(__file__), "shaders\\facet.glsl"))

        # Compile shaders with error checking
        vertex_shader = self.compile_and_check_shader(vertex_shader_code, GL_VERTEX_SHADER)
        geometry_shader = self.compile_and_check_shader(geometry_shader_code, GL_GEOMETRY_SHADER)
        fragment_shader = self.compile_and_check_shader(fragment_shader_code, GL_FRAGMENT_SHADER)

        # Link main rendering program
        self.shader = self.link_and_check_program([vertex_shader, geometry_shader, fragment_shader])

        # Store uniform locations for model, view, projection, and lighting properties
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.viewPos_loc = glGetUniformLocation(self.shader, "viewPos")
        self.projection_loc = glGetUniformLocation(self.shader, "projection")
        self.shininess_loc = glGetUniformLocation(self.shader, "shininess")
        self.specular_intensity_loc = glGetUniformLocation(self.shader, "specularIntensity")
        self.ambient_color_loc = glGetUniformLocation(self.shader, "ambientColor")

        # Retrieve color channel locations if needed for rendering passes
        self.red_channel_loc = glGetUniformLocation(self.shader, "redChannel")
        self.green_channel_loc = glGetUniformLocation(self.shader, "greenChannel")
        self.blue_channel_loc = glGetUniformLocation(self.shader, "blueChannel")

        # Load and compile post-processing shaders: cornea (vertex), retina (fragment), iris (fragment), and ciliary (fragment)
        cornea_code = load_shader_source(os.path.join(os.path.dirname(__file__), "shaders\\cornea.glsl"))
        retina_code = load_shader_source(os.path.join(os.path.dirname(__file__), "shaders\\retina.glsl"))
        iris_code = load_shader_source(os.path.join(os.path.dirname(__file__), "shaders\\iris.glsl"))
        ciliary_code = load_shader_source(os.path.join(os.path.dirname(__file__), "shaders\\ciliary.glsl"))  # Load ciliary shader
        v2_code = load_shader_source(os.path.join(os.path.dirname(__file__), "shaders\\V2.glsl"))  # Load ciliary shader

        # Compile separate cornea shaders for each post-processing stage
        cornea_shader_retina = self.compile_and_check_shader(cornea_code, GL_VERTEX_SHADER)
        cornea_shader_iris = self.compile_and_check_shader(cornea_code, GL_VERTEX_SHADER)
        cornea_shader_v2 = self.compile_and_check_shader(cornea_code, GL_VERTEX_SHADER)
        retina_shader = self.compile_and_check_shader(retina_code, GL_FRAGMENT_SHADER)
        iris_shader = self.compile_and_check_shader(iris_code, GL_FRAGMENT_SHADER)
        self.ciliary_shader = self.compile_and_check_shader(ciliary_code, GL_COMPUTE_SHADER)  # Compile ciliary shader
        self.v2_shader = self.compile_and_check_shader(v2_code, GL_FRAGMENT_SHADER)

        # Link shaders for the retina and iris stages
        self.ciliary_program = self.link_and_check_program([self.ciliary_shader])
        self.v2_program = self.link_and_check_program([cornea_shader_v2, self.v2_shader])
        self.post_processing_shader_one = self.link_and_check_program([cornea_shader_retina, retina_shader])
        self.post_processing_shader_two = self.link_and_check_program([cornea_shader_iris, iris_shader])
        

        # Retrieve and store uniform locations for textures and depth in post-processing shaders
        self.hdr_texture_loc_one = glGetUniformLocation(self.post_processing_shader_one, "hdrBuffer")
        self.depth_texture_loc = glGetUniformLocation(self.post_processing_shader_one, "depthTexture")
        self.hdr_texture_loc_two = glGetUniformLocation(self.post_processing_shader_two, "hdrBuffer")
        self.v2_hdr_texture_loc = glGetUniformLocation(self.v2_program, "hdrBuffer")
        self.v2_depth_texture_loc = glGetUniformLocation(self.v2_program, "distanceBuffer")


        # Retrieve atomic counter locations for aperture control (applied in retina and iris shaders)
        self.old_aperture_counter_loc = glGetUniformLocation(self.post_processing_shader_one, "oldAperture")
        self.new_aperture_counter_loc = glGetUniformLocation(self.post_processing_shader_one, "newAperture")


    def perspective(self, fov, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fov) / 2)
        depth = near - far
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / depth, (2 * far * near) / depth],
            [0, 0, -1, 0]
        ], dtype=np.float32).T  # Note the transpose

    def orthographic(self, left, right, bottom, top, near, far):
        return np.array([
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ], dtype=np.float32).T  # Note the transpose
    
    def set_controller(self, controller):
        self.camera_controller = controller

    def set_perspective_camera(self, camera_pos, forward, up, near, far, view_matrix):
        aspect = self.config.width / self.config.height
        self.projection_matrix = self.perspective(75.0, aspect, near, far)
        self.view_matrix = view_matrix

    def set_orthographic_camera(self, left, right, bottom, top, near, far, camera_pos, forward, up, view_matrix):
        self.projection_matrix = self.orthographic(left, right, bottom, top, near, far)
        self.view_matrix = view_matrix

    def set_flat_camera(self, left, right, bottom, top, near, far, camera_pos, forward, up, view_matrix):
        self.set_orthographic_camera(left, right, bottom, top, near, far, camera_pos, forward, up, view_matrix)
    def generate_phase_buffers(self, colors, n_slices, modulation_type='sinusoidal'):
        """Generate n phase color buffers based on the specified modulation type with minimal memory impact."""
        with torch.no_grad():
            # Move colors to CPU and convert to float16 for reduced memory usage
            colors = colors.clone().detach()
            
            # Prepare phase shifts for each slice
            phase_shifts = 2 * torch.pi * torch.arange(n_slices) / n_slices

            # Initialize phase buffers as int8 with the target shape
            phase_buffers = torch.empty((n_slices, colors.shape[0]), dtype=torch.uint8)
            
            # Generate modulation factors based on type
            if modulation_type == 'sinusoidal':
                factors = torch.sin(phase_shifts).to(dtype=colors.dtype, device=colors.device)
            elif modulation_type == 'linear':
                factors = torch.linspace(0, 1, steps=n_slices).to(dtype=colors.dtype, device=colors.device)
            elif modulation_type == 'none':
                factors = torch.ones(n_slices, dtype=colors.dtype, device=colors.device)
            else:
                raise ValueError(f"Unknown modulation type: {modulation_type}")

            # Normalize colors for modulation
            colors = colors - 0.5

            # Fill phase_buffers by iterating over each factor to reduce intermediate memory allocation
            for i, factor in enumerate(factors):
                # Modulate and fill each slice directly into the pre-allocated buffer
                modulated_colors = (colors * factor + 0.5) * 255
                phase_buffers[i] = modulated_colors.clamp(0, 255).to(torch.uint8)

        return phase_buffers

    def render(self, vertices, indices, normals, color_channels, phase_index=None, diagnostics=True):
        glBindFramebuffer(GL_FRAMEBUFFER, self.deep_fbo)
        glDrawBuffers([self.buffers[0][0]['attachment'], self.buffers[0][1]['attachment'], self.buffers[0][2]['attachment']])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, np.identity(4, dtype=np.float32))
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view_matrix)
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, self.projection_matrix)
        glUniform1f(self.shininess_loc, 1.10)
        glUniform1f(self.specular_intensity_loc, 0.8)
        glUniform4f(self.ambient_color_loc, 1.01, 1.01, 1.01, 1.0)
        glUniform3f(self.viewPos_loc, *self.camera_controller.camera_pos)
                
        self.lighting.set_light_uniforms(self.shader)
        if phase_index is None:
            phase_index = 0
        n_slices = color_channels['red'].shape[0]
        phase_index %= n_slices
        current_red, current_green, current_blue = (
            color_channels['red'][phase_index],
            color_channels['green'][phase_index],
            color_channels['blue'][phase_index]
        )
        self.update_buffers(vertices, indices, normals, current_red, current_green, current_blue)
        self.draw_surface(len(indices))
        self._render_fullscreen_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, self.shallow_fbo)
        glDrawBuffers([self.buffers[1][0]['attachment']])
        glUseProgram(self.v2_program)
        self.check_gl_errors("V2 shader setup")

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.dof_color_texture)
        glUniform1i(self.v2_hdr_texture_loc, 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.dof_distance_texture)
        glUniform1i(self.v2_depth_texture_loc, 1)
        self._render_fullscreen_quad()
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)  # Ensure writes are completed
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(self.ciliary_program)
        self.check_gl_errors("Ciliary shader setup")
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.interest_score_texture)
        glUniform1i(glGetUniformLocation(self.ciliary_program, "interestScoreBuffer"), 2)
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, self.focal_distance_counter_buffer)
        glDispatchCompute(1, 1, 1)
        glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT)  # Ensure counter updates are visible
        glBindFramebuffer(GL_FRAMEBUFFER, self.shallow_fbo)
        glDrawBuffers([self.buffers[1][1]['attachment'],self.buffers[1][3]['attachment']])
        glUseProgram(self.post_processing_shader_one)
        self.check_gl_errors("Retina shader setup")
        glActiveTexture(GL_TEXTURE6)
        glBindTexture(GL_TEXTURE_2D, self.dof_color_texture)
        glUniform1i(self.hdr_texture_loc_one, 6)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, self.dof_distance_texture)
        glUniform1i(self.depth_texture_loc, 3)
        glActiveTexture(GL_TEXTURE5)
        glBindTexture(GL_TEXTURE_2D, self.old_texture)
        glUniform1i(glGetUniformLocation(self.post_processing_shader_one, "oldBuffer"), 5)
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, self.old_aperture_counter_buffer)
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 2, self.new_aperture_counter_buffer)
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 7, self.focal_distance_counter_buffer)
        self._render_fullscreen_quad()
        #glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)  #
        glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT) 
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        selected_texture = self.buffers[self.selected_output[0]].get(self.selected_output[1], {}).get('texture', None)
        if selected_texture:
            glBindFramebuffer(GL_FRAMEBUFFER, self.shallow_fbo)
            glDrawBuffers([self.buffers[1][2]['attachment']])
            glUseProgram(self.post_processing_shader_two)
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, selected_texture)
            glUniform1i(self.hdr_texture_loc_two, 4)
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.new_aperture_counter_buffer)
            glUniform1i(self.new_aperture_counter_loc, 1)
            self.draw_ground_plane()
            self._render_fullscreen_quad()
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.shallow_fbo)
            glReadBuffer(self.buffers[1][2]['attachment'])
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            
            glBlitFramebuffer(
                0, 0, self.config.width, self.config.height,
                0, 0, self.config.width, self.config.height,
                GL_COLOR_BUFFER_BIT, GL_NEAREST
            )


        pygame.display.flip()
        glBindTexture(GL_TEXTURE_2D, 0)  # Unbind after rendering
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Update old aperture for next frame
        glBindBuffer(GL_COPY_READ_BUFFER, self.new_aperture_counter_buffer)
        glBindBuffer(GL_COPY_WRITE_BUFFER, self.old_aperture_counter_buffer)
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, ctypes.sizeof(ctypes.c_uint))
        glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT) 
        # Diagnostics and counter updates (if needed)
        if diagnostics:
            # Retrieve and print counter values
            self.retrieve_and_print_diagnostics()

    def retrieve_and_print_diagnostics(self):
        # Retrieve the focal distance counter value for diagnostics
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.focal_distance_counter_buffer)
        focal_counter = (ctypes.c_uint * 1)()
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, ctypes.sizeof(focal_counter), focal_counter)

        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.new_aperture_counter_buffer)
        aperture_counter = (ctypes.c_uint * 1)()
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, ctypes.sizeof(aperture_counter), aperture_counter)

        # Convert the raw counter values using CounterTranslator
        converted_aperture = CounterTranslator.get_aperture(aperture_counter[0])
        converted_focal_distance = CounterTranslator.get_focal_distance(focal_counter[0])

        # Display diagnostics
        print(f"Aperture Counter: {aperture_counter[0]} (Converted: {converted_aperture:.6f}) | "
            f"Focal Distance Counter: {focal_counter[0]} (Converted: {converted_focal_distance:.6f})")

    def copy_texture(self, src_texture, dst_texture, width, height, copy_depth=False):
        # Generate framebuffers for source and destination
        src_fbo = glGenFramebuffers(1)
        dst_fbo = glGenFramebuffers(1)

        # Ensure the texture IDs are interpreted as GLuints
        src_texture_id = ctypes.c_uint(src_texture)
        dst_texture_id = ctypes.c_uint(dst_texture)

        # Bind the source texture to the source framebuffer with type casting
        glBindFramebuffer(GL_READ_FRAMEBUFFER, src_fbo)
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 if not copy_depth else GL_DEPTH_ATTACHMENT, 
                            GL_TEXTURE_2D, src_texture_id, 0)

        # Bind the destination texture to the destination framebuffer with type casting
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_fbo)
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 if not copy_depth else GL_DEPTH_ATTACHMENT, 
                            GL_TEXTURE_2D, dst_texture_id, 0)

        # Set the buffer bit based on whether we're copying depth or color
        buffer_bit = GL_DEPTH_BUFFER_BIT if copy_depth else GL_COLOR_BUFFER_BIT

        # Blit the contents from source to destination
        glBlitFramebuffer(
            0, 0, width, height,  # Source bounds
            0, 0, width, height,  # Destination bounds
            buffer_bit, GL_NEAREST
        )

        # Clean up by unbinding and deleting framebuffers
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteFramebuffers(1, [src_fbo])
        glDeleteFramebuffers(1, [dst_fbo])
    def check_gl_errors(self, step_description=""):
        error_code = glGetError()
        if error_code != GL_NO_ERROR:
            print(f"OpenGL Error during {step_description}: Code {error_code}")


    def initialize_buffers(self):
        """Initialize all necessary buffers and objects: VAOs, VBOs, EBOs, atomic counters, and HDR DoF framebuffer objects."""
        self.setup_vertex_buffers()
        self.initialize_aperture_counters()
        self.initialize_fullscreen_quad()
        self.initialize_focal_distance_counter()
        return self.initialize_main_buffers()



    def setup_vertex_buffers(self):
        """Set up Vertex Array Object (VAO), Vertex Buffer Objects (VBOs), and Element Buffer Object (EBO)."""
        # Generate and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Generate VBO for positions
        self.vbo_positions = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)  # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # Generate VBO for red color component
        self.vbo_red = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_red)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(1)  # Red attribute
        glVertexAttribPointer(1, 1, GL_UNSIGNED_BYTE, GL_TRUE, 0, ctypes.c_void_p(0))

        # Generate VBO for green color component
        self.vbo_green = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_green)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(2)  # Green attribute
        glVertexAttribPointer(2, 1, GL_UNSIGNED_BYTE, GL_TRUE, 0, ctypes.c_void_p(0))

        # Generate VBO for blue color component
        self.vbo_blue = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_blue)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(3)  # Blue attribute
        glVertexAttribPointer(3, 1, GL_UNSIGNED_BYTE, GL_TRUE, 0, ctypes.c_void_p(0))

        # Generate VBO for normals
        self.vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(4)  # Normal attribute
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # Generate EBO for indices
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)

        # Unbind VAO and VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
    def initialize_focal_distance_counter(self):
        """Initialize the atomic counter buffer for focal distance tracking."""
        # Create atomic counter buffer for focal distance
        self.focal_distance_counter_buffer = glGenBuffers(1)
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.focal_distance_counter_buffer)

        # Set an initial value for the counter
        initial_value = np.array([100000000], dtype=np.uint32)
        #initial_value = np.array([2000000000], dtype=np.uint32)
        glBufferData(GL_ATOMIC_COUNTER_BUFFER, initial_value.nbytes, initial_value, GL_DYNAMIC_DRAW)

        # Unbind the buffer after setup
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)


    def initialize_aperture_counters(self):
        """Initialize atomic counter buffers for aperture tracking."""
        # Initialize old aperture atomic counter buffer
        self.old_aperture_counter_buffer = glGenBuffers(1)
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.old_aperture_counter_buffer)
        old_aperture_initial_value = np.array([3000000000], dtype=np.uint32)
        glBufferData(GL_ATOMIC_COUNTER_BUFFER, old_aperture_initial_value.nbytes, old_aperture_initial_value, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

        # Initialize new aperture atomic counter buffer
        self.new_aperture_counter_buffer = glGenBuffers(1)
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, self.new_aperture_counter_buffer)
        new_aperture_initial_value = np.array([3000000000], dtype=np.uint32)
        glBufferData(GL_ATOMIC_COUNTER_BUFFER, new_aperture_initial_value.nbytes, new_aperture_initial_value, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

    def initialize_test_buffer(self):
        return

    def initialize_main_buffers(self):
        """Initialize main framebuffer with all texture attachments."""
        # Create main framebuffer
        return_buffers = []
        for j, buffer_set in enumerate(self.buffers):
            buffer_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, buffer_fbo)

            # Attach each texture from the combined structure to the framebuffer
            for idx, buffer_info in buffer_set.items():
                # Create a blank tensor (texture data) with ones
                zero_data = np.ones((self.config.height, self.config.width, 4), dtype=np.float32)
                n, m = 10*(idx+1), 15*(j+1)  # Modify n and m as needed for line density
                for y in range(zero_data.shape[0]):
                    for x in range(zero_data.shape[1]):
                        if y % n == 0 or x % m == 0:
                            grey_value = np.random.uniform(0.3, 0.7)  # Random grey between 0.3 and 0.7
                            zero_data[y, x, :3] = grey_value  # Set RGB channels to grey_value
                texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.config.width, self.config.height, 0, GL_RGBA, GL_FLOAT, zero_data)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glFramebufferTexture2D(GL_FRAMEBUFFER, buffer_info['attachment'], GL_TEXTURE_2D, texture, 0)
                buffer_info['texture'] = texture  # Assign generated texture ID
                setattr(self, buffer_info['name'], texture)
            
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("Main Framebuffer not complete")
                glBindFramebuffer(GL_FRAMEBUFFER, 0)  
            return_buffers.append(buffer_fbo)
            
            self.activate_all_draw_buffers(buffer_fbo, buffer_set)

        # Depth renderbuffer for depth testing
        glBindFramebuffer(GL_FRAMEBUFFER, return_buffers[0])
        self.main_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.main_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.config.width, self.config.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.main_rbo)
        # Check framebuffer completeness


        return return_buffers[0], return_buffers[1]
        
    def initialize_fullscreen_quad(self):
        """Initialize and store a persistent VBO for a fullscreen quad."""
        quad_vertices = np.array([
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
            1.0,  1.0, 0.0,
        ], dtype=np.float32)

        # Generate and bind a VBO for fullscreen quad
        self.quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        
        # Unbind buffer to prevent accidental changes
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_buffers_opengl(self, vbo_id=None, texture_id=None):
        if vbo_id:
            glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
            # Set up vertex attribute pointers as needed
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        if texture_id:
            glBindTexture(GL_TEXTURE_2D, texture_id)
            # Bind texture for subsequent rendering
            glBindTexture(GL_TEXTURE_2D, 0)

    def update_buffers(self, vertices, indices, normals, red, green, blue):
        """
        Update the data in the existing buffers.
        
        Parameters:
        - vertices: torch.Tensor of vertex positions (shape: [num_vertices, 3]).
        - indices: torch.Tensor of indices (shape: [num_indices]).
        - normals: torch.Tensor of normals (shape: [num_vertices, 3]).
        - red: torch.Tensor of red color components (shape: [num_vertices]).
        - green: torch.Tensor of green color components (shape: [num_vertices]).
        - blue: torch.Tensor of blue color components (shape: [num_vertices]).
        """
        # Ensure data is on CPU and convert to NumPy
        vertices_np = vertices.clone().detach().cpu().numpy().astype(np.float32)
        indices_np = indices.clone().detach().cpu().numpy()
        normals_np = normals.clone().detach().cpu().numpy().astype(np.float32)
        red_np = red.clone().detach().cpu().numpy()
        green_np = green.clone().detach().cpu().numpy()
        blue_np = blue.clone().detach().cpu().numpy()

        # Update positions
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL_DYNAMIC_DRAW)

        # Update red
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_red)
        glBufferData(GL_ARRAY_BUFFER, red_np.nbytes, red_np, GL_DYNAMIC_DRAW)

        # Update green
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_green)
        glBufferData(GL_ARRAY_BUFFER, green_np.nbytes, green_np, GL_DYNAMIC_DRAW)

        # Update blue
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_blue)
        glBufferData(GL_ARRAY_BUFFER, blue_np.nbytes, blue_np, GL_DYNAMIC_DRAW)

        # Update normals
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, normals_np.nbytes, normals_np, GL_DYNAMIC_DRAW)

        # Update indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_np.nbytes, indices_np, GL_DYNAMIC_DRAW)

        # Unbind buffers
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw_surface(self, indices_count):
        """
        Draw the surface using the pre-initialized VAO and EBO.
        
        Parameters:
        - indices_count: Number of indices to draw.
        """
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glDrawElements(GL_TRIANGLE_STRIP, indices_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)




    def _render_fullscreen_quad(self):
        """
        Render a fullscreen quad using a persistent VBO.
        """
        # Bind the pre-initialized VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        
        # Enable and bind position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # Draw quad with two triangles
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        #glDrawArrays(GL_POINTS, 0, 4)

        # Cleanup (unbind but don’t delete the VBO)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)