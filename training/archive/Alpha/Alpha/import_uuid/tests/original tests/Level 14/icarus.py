import pymunk
import threading
import time
import queue
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import random
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

class ChipmunkSlice:
    def __init__(self):
        # Simulation space
        self.space = pymunk.Space()
        self.space.gravity = (0, -981)  # Gravity in negative Y direction

        # Dynamic body
        self.body = pymunk.Body(mass=1, moment=10)
        self.body.position = (0, 10)  # Starting position
        self.shape = pymunk.Circle(self.body, radius=1)
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
        self.dt = 0.001  # Default time step duration

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

    def generate_training_data(self, duration, dt, state_dict=None, randomize_missing=True, batch_size=1):
        """
        Generate training data by simulating the body with given or random initial conditions.

        Parameters:
        - duration: Total time to simulate.
        - dt: Time step.
        - state_dict: Dictionary specifying initial conditions (may be partial). Keys can include:
            - 'velocity': (vx, vy)
            - 'force_vector': (fx, fy)
            - 'mass': m
            - 'moment': moment
            - 'torque': torque
            - 'angular_velocity': av
            - 'wall_properties': {...}
        - randomize_missing: Whether to randomize unspecified parameters. If False, use defaults.
        - batch_size: Number of simulations to run.

        Returns:
        - A list of dictionaries with 'input' and 'output' keys for training.
        """
        data = []
        for _ in range(batch_size):
            # Reset the simulation
            self.reset_simulation()
            # Set initial position to (0, 0)
            

            # Prepare state_dict
            if state_dict is None:
                state_dict = {}

            inputs = {}

            # Randomize missing parameters if required
            if randomize_missing:
                if 'position' not in state_dict:
                    self.body.position = (random.uniform(-7, 7), random.uniform(3,17))
                    state_dict['position'] = self.body.position
                if 'velocity' not in state_dict:
                    vx = random.uniform(-10, 10)
                    vy = random.uniform(-10, 10)
                    state_dict['velocity'] = (vx, vy)
                if 'force_vector' not in state_dict:
                    fx = random.uniform(-1000, 1000)
                    fy = random.uniform(-1000, 1000)
                    state_dict['force_vector'] = (fx, fy)
                if 'mass' not in state_dict:
                    mass = random.uniform(1, 10)
                    state_dict['mass'] = mass
                # Randomize other parameters as needed
            else:
                # Use default values for missing parameters
                if 'position' not in state_dict:
                    self.body.position = (0, 10)
                    state_dict['position'] = self.body.position
                if 'velocity' not in state_dict:
                    state_dict['velocity'] = (0, 0)
                if 'force_vector' not in state_dict:
                    state_dict['force_vector'] = (0, 0)
                if 'mass' not in state_dict:
                    state_dict['mass'] = 1
                # Set other parameters to defaults as needed

            # Apply inputs
            self.apply_inputs(state_dict)

            # Record initial state
            initial_state = [
                self.body.position[0],  # position_x
                self.body.position[1],  # position_y
                self.body.velocity[0],  # velocity_x
                self.body.velocity[1],  # velocity_y
                state_dict.get('force_vector', (0, 0))[0],  # force_x
                state_dict.get('force_vector', (0, 0))[1],  # force_y
                self.body.mass  # mass
            ]

            # Simulate for duration
            num_steps = int(duration / dt)
            original_dt = self.dt
            self.dt = dt  # Update the time step
            for _ in range(num_steps):
                self.simulation_step()
            self.dt = original_dt  # Restore original dt

            # Record final state
            final_position = self.body.position
            final_velocity = self.body.velocity

            output_state = [
                final_position[0],  # next_position_x
                final_position[1],  # next_position_y
                final_velocity[0],  # next_velocity_x
                final_velocity[1]   # next_velocity_y
            ]

            data.append({'input': initial_state, 'output': output_state})

        return data

class SimpleNN(nn.Module):
    def __init__(self, input_size=7, output_size=4):
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
    input_size = 7  # position_x, position_y, velocity_x, velocity_y, force_x, force_y, mass
    output_size = 4  # next_position_x, next_position_y, next_velocity_x, next_velocity_y
    net = SimpleNN(input_size, output_size)

    # Autosave and Autoload the model
    model_path = 'model.pth'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        print("Model loaded from", model_path)
    else:
        print("No existing model found. Starting fresh.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Create a ChipmunkSlice instance
    chipmunk_slice = ChipmunkSlice()

    # Training parameters
    num_epochs = 5  # Adjust as needed
    batch_size = 640  # Adjust as needed
    dt = chipmunk_slice.dt  # Use the default dt from ChipmunkSlice

    # Load existing dataset or create a new one
    dataset_path = 'dataset.pt'
    if os.path.exists(dataset_path):
        data = torch.load(dataset_path)
        data_inputs = data['inputs']
        data_targets = data['targets']
        print("Dataset loaded from", dataset_path)
    else:
        data_inputs = torch.empty((0, input_size))
        data_targets = torch.empty((0, output_size))
        print("No existing dataset found. Starting data collection.")

    # Collect new training data using generate_training_data
    num_new_samples = 10000  # Number of new samples to collect
    print("Collecting new training data...")
    batch_size_data_gen = 1000  # Batch size for data generation
    num_batches = num_new_samples // batch_size_data_gen

    new_inputs = []
    new_targets = []

    for _ in range(num_batches):
        # Generate data
        data_batch = chipmunk_slice.generate_training_data(
            duration=dt,  # Simulate for one time step
            dt=dt,
            state_dict=None,  # No initial state specified
            randomize_missing=True,
            batch_size=batch_size_data_gen
        )

        # Extract inputs and outputs
        inputs_batch = [sample['input'] for sample in data_batch]
        targets_batch = [sample['output'] for sample in data_batch]

        new_inputs.extend(inputs_batch)
        new_targets.extend(targets_batch)

    # Convert to tensors
    new_inputs = torch.tensor(new_inputs, dtype=torch.float32)
    new_targets = torch.tensor(new_targets, dtype=torch.float32)

    # Append new data to existing data
    data_inputs = torch.cat((data_inputs, new_inputs), dim=0)
    data_targets = torch.cat((data_targets, new_targets), dim=0)

    # Save the updated dataset
    torch.save({'inputs': data_inputs, 'targets': data_targets}, dataset_path)
    print("Dataset saved to", dataset_path)

    # Prepare data loaders
    train_dataset = torch.utils.data.TensorDataset(data_inputs, data_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print("Training the neural network...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs_batch, targets_batch in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs_batch)

            # Compute the loss
            loss = criterion(outputs, targets_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Report the loss
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    print("Training complete.")

    # Save the trained model
    torch.save(net.state_dict(), model_path)
    print("Model saved to", model_path)

    # Simulation using the neural network with multiple balls
    print("Running the NN-based simulation with multiple balls...")

    # Initialize Pygame
    pygame.init()
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("NN Simulated Positions - Multiple Balls")
    clock = pygame.time.Clock()

    # Simulation parameters
    num_balls = 100  # Number of balls to simulate
    mass = torch.ones(num_balls) * 5.0  # Mass of each ball

    # Initial states
    positions = torch.zeros((num_balls, 2))  # Positions start at (0, 0)
    positions = positions + torch.tensor([0, 10])

    velocities = (torch.rand((num_balls, 2)) - 0.5) * 20  # Random velocities between -10 and 10
    forces = torch.zeros((num_balls, 2))
    masses = mass.unsqueeze(1)

    # State tensor: [position_x, position_y, velocity_x, velocity_y, force_x, force_y, mass]
    states = torch.cat((positions, velocities, forces, masses), dim=1)

    # Simulation loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Apply random forces occasionally
        random_forces = (torch.rand((num_balls, 2)) - 0.5) * 10  # Random forces between -500 and 500
        random_mask = (torch.rand(num_balls) < 0.1).unsqueeze(1)
        forces = random_mask * random_forces
        states[:, 4:6] = forces

        # Prepare input tensor
        input_tensor = states.clone()

        # Predict next state
        with torch.no_grad():
            outputs = net(input_tensor)

        # Update the states
        states[:, 0:4] = outputs  # Update positions and velocities
        states[:, 6] = mass  # Keep mass constant

        # Clear screen
        screen.fill((255, 255, 255))

        # Convert positions to screen coordinates
        screen_positions = states[:, 0:2].numpy()
        x_coords = (screen_positions[:, 0] + 10) * (screen_width / 20)
        y_coords = screen_height - (screen_positions[:, 1]) * (screen_height / 20)
        print(x_coords)
        print(y_coords)
        # Draw the balls using vectorized operations
        for x, y in zip(x_coords, y_coords):
            pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 5)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()
