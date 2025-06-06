import math
import numpy as np
import pygame
import torch
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

###############################################
# Configuration
###############################################
WIN_WIDTH, WIN_HEIGHT = 800, 600

# Domain configuration
X_RANGE = (-10, 10)
Y_RANGE = (-10, 10)
Z_RANGE = (-10, 10)

DT = 0.01
MAX_VELOCITY = 1.0

# Define prime numbers for vertices identification
PRIMES = [2, 3, 5, 7, 11, 13]  # Assign a prime per vertex for demonstration

###############################################
# Domain and Boundary
###############################################
class Domain3D:
    def __init__(self, x_range, y_range, z_range, mode='reflect'):
        self.xmin, self.xmax = x_range
        self.ymin, self.ymax = y_range
        self.zmin, self.zmax = z_range
        self.mode = mode

    def apply_boundaries(self, pos, vel):
        x, y, z = pos
        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax or z < self.zmin or z > self.zmax:
            # Simple reflect mode
            if x < self.xmin or x > self.xmax:
                vel[0] = -vel[0]
                pos[0] = max(min(x, self.xmax), self.xmin)
            if y < self.ymin or y > self.ymax:
                vel[1] = -vel[1]
                pos[1] = max(min(y, self.ymax), self.ymin)
            if z < self.zmin or z > self.zmax:
                vel[2] = -vel[2]
                pos[2] = max(min(z, self.zmax), self.zmin)
        return pos, vel

###############################################
# Shuttle (One shuttle with multiple vertices)
###############################################
class Shuttle:
    def __init__(self, vertices, mass=1.0):
        # vertices: [(x,y,z), ...]
        self.vertices = np.array(vertices, dtype=float)
        self.velocities = np.zeros_like(self.vertices)
        self.mass = mass
        self.accelerations = np.zeros_like(self.vertices)
        self.alive = True

    def apply_forces(self, forces):
        # forces: same shape as vertices
        net_force = np.sum(forces, axis=0) # sum all vertices force to get a shuttle-level acceleration if needed
        # Alternatively, apply per-vertex acceleration:
        self.accelerations = forces / self.mass

    def integrate(self, dt, max_velocity):
        # Integrate per-vertex
        for i in range(len(self.vertices)):
            self.velocities[i] += self.accelerations[i]*dt
            speed = np.linalg.norm(self.velocities[i])
            if speed > max_velocity:
                self.velocities[i] = (self.velocities[i]/speed)*max_velocity
            self.vertices[i] += self.velocities[i]*dt

###############################################
# Mass Field Definition (Piecewise spherical domains)
###############################################
# Define a chain of lambdas representing spherical mass contributions
# Each lambda returns a scalar contribution if pos is inside a sphere
# piecewise_spheres = [(center, radius, value), ...]
piecewise_spheres = [
    ((0.0,0.0,0.0), 2.0, 1.0),    # Sphere at origin
    ((2.0,2.0,2.0), 1.0, 0.5)     # Another smaller sphere
]

def mass_field_fn_chain(pos):
    # pos: (x,y,z)
    val = 0.0
    for (cx,cy,cz), r, v in piecewise_spheres:
        dx,dy,dz = pos[0]-cx, pos[1]-cy, pos[2]-cz
        dist = math.sqrt(dx*dx+dy*dy+dz*dz)
        if dist < r:
            val += v
    return val

###############################################
# Binary Decision Tree Logic for Collision Response
###############################################
# Suppose if mass_field_val > 1.0 => collision overlap
# if overlap: if mass_field_val > 2.0 vanish else reflect
def collision_decision(mass_val):
    # Binary logic:
    # mass_val <= 1.0: no collision
    # 1.0 < mass_val <= 2.0: reflect scenario
    # mass_val > 2.0: vanish scenario
    if mass_val <= 1.0:
        return 'none'
    elif mass_val <= 2.0:
        return 'reflect'
    else:
        return 'vanish'

###############################################
# Prime-sum based vertex activation
###############################################
def vertex_activation(shuttle):
    # For each vertex, check mass field and sum prime if inside
    # If sum > 0 => active
    active_mask = []
    for i, vpos in enumerate(shuttle.vertices):
        val = mass_field_fn_chain(vpos)
        # if inside any sphere val>0
        # use prime sums: if val>0 add PRIMES[i]
        vsum = 0
        if val > 0:
            vsum += PRIMES[i % len(PRIMES)]
        # If vsum>0 active
        active_mask.append(vsum>0)
    return active_mask

###############################################
# GPU Integration (Minimal)
###############################################
shader_code = """
#version 420 core
layout(local_size_x = 128) in;

layout(std430, binding=0) buffer VertexPos {
    vec4 vertex_positions[];
};
layout(std430, binding=1) buffer MassItems {
    vec4 mass_positions[]; 
};
layout(std430, binding=2) buffer Forces {
    vec4 output_forces[];
};

uniform uint num_vertices;
uniform uint num_mass_items;

void main(){
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_vertices) return;

    vec3 vpos = vertex_positions[idx].xyz;
    vec3 total_force = vec3(0.0);

    for (uint i=0; i<num_mass_items; i++){
        vec3 mpos = mass_positions[i].xyz;
        vec3 dir = mpos - vpos;
        float dist = length(dir);
        if (dist > 0.0001) {
            // simple attraction
            vec3 f = normalize(dir)*(0.01/(dist*dist));
            total_force += f;
        }
    }

    output_forces[idx] = vec4(total_force, 0.0);
}
""".strip()

def init_opengl():
    if not glfw.init():
        raise Exception("GLFW init failed")
    window = glfw.create_window(1,1,"Offscreen",None,None)
    glfw.make_context_current(window)
    return window

###############################################
# Main Simulation
###############################################
def main():
    # Initialize OpenGL
    window = init_opengl()
    program = compileProgram(compileShader(shader_code, GL_COMPUTE_SHADER))

    # Initialize pygame for visualization
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    # Single shuttle with multiple vertices
    # Let's do 3 vertices forming a small triangle
    shuttle = Shuttle(vertices=[(-0.5,0,0),(0.5,0,0),(0,0.5,0)])

    # Mass items (just random points)
    mass_positions = np.array([[0.0,0.0,1.0,1.0],
                               [2.0,2.0,2.0,1.0]], dtype=np.float32)
    num_mass_items = mass_positions.shape[0]

    # OpenGL Buffers
    # Vertex buffer
    vertex_positions = np.array([list(v)+[1.0] for v in shuttle.vertices], dtype=np.float32)
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertex_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, vertex_positions.nbytes, vertex_positions, GL_DYNAMIC_DRAW)

    mass_buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mass_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, mass_positions.nbytes, mass_positions, GL_DYNAMIC_DRAW)

    forces_buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, forces_buffer)
    forces_array = np.zeros((len(shuttle.vertices),4), dtype=np.float32)
    glBufferData(GL_SHADER_STORAGE_BUFFER, forces_array.nbytes, forces_array, GL_DYNAMIC_DRAW)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertex_buffer)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mass_buffer)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, forces_buffer)

    # Uniform setup
    glUseProgram(program)
    glUniform1ui(glGetUniformLocation(program, "num_vertices"), len(shuttle.vertices))
    glUniform1ui(glGetUniformLocation(program, "num_mass_items"), num_mass_items)

    domain = Domain3D(X_RANGE, Y_RANGE, Z_RANGE, mode='reflect')

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running=False

        if not shuttle.alive:
            # Reinitialize if vanished
            shuttle = Shuttle(vertices=[(-0.5,0,0),(0.5,0,0),(0,0.5,0)])

        # CPU: Check vertex activation
        active_mask = vertex_activation(shuttle)
        # sum of active vertices prime codes can be used to identify collision states
        # Just a demonstration:
        sum_primes = 0
        for i,a in enumerate(active_mask):
            if a:
                sum_primes += PRIMES[i%len(PRIMES)]
        # Use mass_field_fn_chain on centroid to decide collision
        centroid = np.mean(shuttle.vertices, axis=0)
        mass_val = mass_field_fn_chain(centroid)
        decision = collision_decision(mass_val)
        if decision == 'vanish':
            shuttle.alive = False
        elif decision == 'reflect':
            # Just invert velocities
            shuttle.velocities = -shuttle.velocities

        # GPU pass: Upload current vertex positions
        vertex_positions = np.array([list(v)+[1.0] for v in shuttle.vertices], dtype=np.float32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertex_buffer)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vertex_positions.nbytes, vertex_positions)

        # Compute shader dispatch
        glUseProgram(program)
        glDispatchCompute((len(shuttle.vertices)//128)+1,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Retrieve forces
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, forces_buffer)
        result = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, forces_array.nbytes)
        updated_forces = np.frombuffer(result, dtype=np.float32).reshape(len(shuttle.vertices),4)

        # Apply forces
        shuttle.apply_forces(updated_forces[:,:3])

        # Integrate
        shuttle.integrate(DT, MAX_VELOCITY)

        # Apply domain boundaries
        for i in range(len(shuttle.vertices)):
            shuttle.vertices[i], shuttle.velocities[i] = domain.apply_boundaries(shuttle.vertices[i], shuttle.velocities[i])

        # Visualization
        screen.fill((0,0,0))
        for v in shuttle.vertices:
            x2d = int(WIN_WIDTH//2 + v[0]*20)
            y2d = int(WIN_HEIGHT//2 - v[1]*20)
            color = (0,255,0) if shuttle.alive else (255,0,0)
            pygame.draw.circle(screen, color, (x2d, y2d), 5)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    glfw.terminate()

if __name__ == "__main__":
    main()
