import pymunk
import threading
import time
import queue
import logging
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

class ChipmunkSlice:
    def __init__(self):
        # Simulation space
        self.space = pymunk.Space()
        #self.space.gravity = (0, -9.81)  # Gravity in negative Y direction

        # Dynamic body
        self.body = pymunk.Body(mass=1, moment=10)
        self.body.position = (0, 10)  # Starting position
        self.shape = pymunk.Circle(self.body, radius=.1)
        self.space.add(self.body, self.shape)

        # Walls (static bodies)
        self.walls = []
        self.create_walls()

        # Queues for input and state
        self.input_queue = queue.Queue()
        self.state_log_queue = queue.Queue()
        self.collision_log_queue = queue.deque(maxlen=1000)  # Rolling queue for state changes

        # Simulation control
        self.running = False
        self.simulation_thread = None

        # Simulation mode
        self.active_mode = True  # True for live mode, False for manual step
        self.reset_position_each_step = False  # Flag to reset position each step

        # Persistent tracking
        self.velocity_integral = (0, 0)  # Local translation tracking
        self.temperature = 20.0  # Placeholder for external management

        # Collision detection
        self.in_collision = False
        self.collision_handler = self.space.add_default_collision_handler()
        self.collision_handler.begin = self.on_collision_begin
        self.collision_handler.separate = self.on_collision_separate

        # Time step
        self.dt = 0.1  # Default time step duration

    def create_walls(self):
        """
        Create walls around the simulation area with default properties.
        """
        static_body = self.space.static_body
        walls = [
            pymunk.Segment(static_body, (-10, 0), (-10, 20), 1),  # Left wall
            pymunk.Segment(static_body, (10, 0), (10, 20), 1),    # Right wall
            pymunk.Segment(static_body, (-10, 0), (10, 0), 1),    # Ground
            pymunk.Segment(static_body, (-10, 20), (10, 20), 1),  # Ceiling
        ]
        for wall in walls:
            wall.friction = 0.9
            wall.elasticity = 0.9
            self.space.add(wall)
            self.walls.append(wall)

    def apply_inputs(self, inputs):
        """
        Apply inputs from a dictionary to the simulation.
        """
        if "force_vector" in inputs:
            force = inputs["force_vector"]
            self.body.apply_force_at_local_point(force)

        if "velocity" in inputs:
            velocity = inputs["velocity"]
            self.body.velocity = velocity

        if "mass" in inputs:
            mass = inputs["mass"]
            self.body.mass = mass

        if "moment" in inputs:
            moment = inputs["moment"]
            self.body.moment = moment

        if "torque" in inputs:
            torque = inputs["torque"]
            self.body.apply_force_at_local_point((0, torque))

        if "angular_velocity" in inputs:
            angular_velocity = inputs["angular_velocity"]
            self.body.angular_velocity = angular_velocity

        if "dt_granularity_adjustment" in inputs:
            new_dt = inputs["dt_granularity_adjustment"]
            self.dt = new_dt
            self.state_log_queue.put(f"Granularity adjusted to: {new_dt}")

        if "radiation" in inputs:
            radiation = inputs["radiation"]
            # Handle radiation (external management)

        if "energy" in inputs:
            energy = inputs["energy"]
            # Handle energy (temperature change)

        if "wall_properties" in inputs:
            properties = inputs["wall_properties"]
            for wall in self.walls:
                wall.friction = properties.get("friction", wall.friction)
                wall.elasticity = properties.get("elasticity", wall.elasticity)

    def simulation_step(self):
        """
        Perform a single simulation step.
        """
        while not self.input_queue.empty():
            inputs = self.input_queue.get()
            self.apply_inputs(inputs)

        self.space.step(self.dt)

        # Update local integrals
        self.velocity_integral = (
            self.velocity_integral[0] + self.body.velocity[0] * self.dt,
            self.velocity_integral[1] + self.body.velocity[1] * self.dt,
        )

        # Log state
        state = {
            "position": self.body.position,
            "velocity": self.body.velocity,
            "time": time.time()
        }
        self.state_log_queue.put(state)

        # Rolling log of state changes
        self.collision_log_queue.append(state)

        # Reset position if flag is set
        if self.reset_position_each_step:
            self.body.position = (0, 10)

    def run_simulation(self):
        """
        Run simulation in active mode.
        """
        self.running = True
        start_time = time.time()
        while self.running:
            self.simulation_step()
            time.sleep(self.dt)

    def manual_step(self):
        """
        Perform a single step in manual mode.
        """
        self.simulation_step()

    def start_simulation(self):
        """
        Start the simulation in active mode.
        """
        if self.simulation_thread is None:
            self.simulation_thread = threading.Thread(target=self.run_simulation)
            self.simulation_thread.start()

    def stop_simulation(self):
        """
        Stop the simulation.
        """
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join()
            self.simulation_thread = None

    def reset_simulation(self):
        """
        Reset the simulation to its initial state.
        """
        self.body.position = (0, 10)
        self.body.velocity = (0, 0)
        self.body.force = (0, 0)
        self.body.torque = 0
        self.body.angular_velocity = 0
        self.velocity_integral = (0, 0)
        self.in_collision = False
        self.collision_log_queue.clear()

    def on_collision_begin(self, arbiter, space, data):
        """
        Collision handler when collision begins.
        """
        self.in_collision = True
        contact_point_set = arbiter.contact_point_set
        points = [point.point_a for point in contact_point_set.points]
        collision_info = {
            "collision_begin_time": time.time(),
            "collision_points": points,
            "body_velocity": self.body.velocity
        }
        self.state_log_queue.put(collision_info)
        return True  # Continue processing collision

    def on_collision_separate(self, arbiter, space, data):
        """
        Collision handler when collision ends.
        """
        self.in_collision = False
        # Dump the collision log
        while self.collision_log_queue:
            state = self.collision_log_queue.popleft()
            self.state_log_queue.put(state)
        self.state_log_queue.put({
            "collision_end_time": time.time(),
            "body_position": self.body.position
        })

class SimpleNN(nn.Module):
    def __init__(self, input_size=3, output_size=2):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    # Initialize the neural network
    input_size = 3  # force_x, force_y, mass
    output_size = 2  # position_x, position_y
    net = SimpleNN(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Create a ChipmunkSlice instance
    chipmunk_slice = ChipmunkSlice()

    # Training loop
    num_epochs = 1000000
    for epoch in range(num_epochs):
        # Reset the simulation
        chipmunk_slice.reset_simulation()

        # Generate random inputs
        force_x = (torch.rand(1).item() - 0.5) * 20000  # Random force between -1000 and 1000
        force_y = (torch.rand(1).item() - 0.5) * 20000
        mass = torch.rand(1).item() * 10 + 1  # Random mass between 1 and 11

        inputs = {
            "force_vector": (force_x, force_y),
            "mass": mass
        }

        # Apply inputs to the simulation
        chipmunk_slice.apply_inputs(inputs)

        # Run the simulation for one second
        sim_duration = 0.2
        num_steps = int(sim_duration / chipmunk_slice.dt)
        for _ in range(num_steps):
            chipmunk_slice.simulation_step()

        # Get the end position
        end_position = chipmunk_slice.body.position
        position_x = end_position[0]
        position_y = end_position[1]

        # Prepare the inputs and outputs as tensors
        input_tensor = torch.tensor([force_x, force_y, mass], dtype=torch.float32).unsqueeze(0)  # Shape [1, 3]
        target_tensor = torch.tensor([position_x, position_y], dtype=torch.float32).unsqueeze(0)  # Shape [1, 2]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = net(input_tensor)

        # Compute the loss
        loss = criterion(output, target_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Report the loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
