import torch
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import asyncio

# ------------------------------
# GLSL Shader Program Loader
# ------------------------------
def load_compute_shader(shader_path):
    with open(shader_path, "r") as file:
        shader_code = file.read()
    return compileProgram(compileShader(shader_code, GL_COMPUTE_SHADER))

# ------------------------------
# GPU Buffer Management
# ------------------------------
def upload_tensor_to_gpu(tensor):
    buffer_id = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, tensor.numpy(), GL_DYNAMIC_DRAW)
    return buffer_id

def download_tensor_from_gpu(buffer_id, tensor_shape, dtype):
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    result = np.zeros(tensor_shape, dtype=np.float32)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, result.nbytes, result)
    return torch.tensor(result, dtype=dtype)

# ------------------------------
# Dynamic Lambda Chain
# ------------------------------
def construct_lambda_chain(chain_string):
    """Dynamically assembles a chain of lambdas from a string."""
    operations = {
        "increment": lambda x: x + 1,
        "double": lambda x: x * 2,
        "square": lambda x: x ** 2,
    }
    ops = chain_string.split("->")
    func_chain = [operations[op.strip()] for op in ops]
    return lambda x: torch.stack([func(x) for func in func_chain]).sum(dim=0)

# ------------------------------
# Shuttle Fleet Manager
# ------------------------------
class ShuttleFleet:
    def __init__(self, fleet_tensor, shader_path, lambda_chain_string):
        self.fleet_tensor = fleet_tensor  # Fleet data tensor
        self.shader = load_compute_shader(shader_path)
        self.lambda_chain = construct_lambda_chain(lambda_chain_string)
        self.buffer_id = None

    async def execute_shader(self):
        """Run GLSL compute shader."""
        glUseProgram(self.shader)
        self.buffer_id = upload_tensor_to_gpu(self.fleet_tensor)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.buffer_id)
        glDispatchCompute((self.fleet_tensor.shape[0] + 127) // 128, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    async def execute_lambda_chain(self):
        """Execute training mode lambda chain on the fleet tensor."""
        async def train_shuttle(shuttle):
            return self.lambda_chain(shuttle)

        results = await asyncio.gather(*[train_shuttle(shuttle) for shuttle in self.fleet_tensor])
        self.fleet_tensor = torch.stack(results)

    async def blend_results(self):
        """Blend GPU and CPU results."""
        gpu_results = download_tensor_from_gpu(self.buffer_id, self.fleet_tensor.shape, self.fleet_tensor.dtype)
        self.fleet_tensor = (gpu_results + self.fleet_tensor) / 2.0  # Simple averaging blend

    async def run(self):
        """Run full shuttle lifecycle."""
        await asyncio.gather(self.execute_shader(), self.execute_lambda_chain())
        await self.blend_results()
        print("Final Fleet Tensor:")
        print(self.fleet_tensor)

# ------------------------------
# Main Execution
# ------------------------------
from OpenGL.GL import *
import glfw

def main():
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW failed to initialize!")

    # Set GLFW window hints for OpenGL version 4.2 compatibility
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Invisible window for compute shaders

    # Create an OpenGL context/window
    window = glfw.create_window(800, 600, "Shuttle Fleet", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed!")

    # Make the OpenGL context current
    glfw.make_context_current(window)

    # Load OpenGL extensions (if needed)
    import OpenGL.GL.shaders  # Ensure shader functions are available

    print("OpenGL context successfully initialized!")
    print(f"OpenGL version: {glGetString(GL_VERSION).decode('utf-8')}")

    # Initialize fleet tensor
    fleet_tensor = torch.zeros((1024, 4), dtype=torch.float32, device="cpu")

    # Shader file path
    shader_path = "shaders/minimal_op.glsl"
    lambda_chain_string = "increment -> double -> square"

    # Run the Shuttle Fleet
    fleet = ShuttleFleet(fleet_tensor, shader_path, lambda_chain_string)
    asyncio.run(fleet.run())

    # Terminate GLFW (cleanup)
    glfw.terminate()

if __name__ == "__main__":
    main()
