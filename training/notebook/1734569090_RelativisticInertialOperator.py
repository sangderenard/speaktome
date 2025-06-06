import torch

class RelativisticInertialOperator:
    """
    Relativistic Inertial Operator for performing discrete integration and derivation
    of motion across position, velocity, momentum, and higher-order derivatives.
    """

    def __init__(self, fields, params):
        """
        Initialize the operator with configured fields and engine parameters.

        Args:
            fields (list): Fields to track in the state tensor (e.g., position, velocity).
            params (dict): Engine configuration parameters.
                - base_mass: Default mass of objects (kg).
                - base_moment: Default moment of inertia for angular motion.
                - speed_of_light: Speed of light in m/s.
                - velocity_threshold: Threshold fraction of c to apply relativistic corrections.
                - history_length: Maximum dt history length for accurate updates.
        """
        self.fields = fields
        self.params = params
        self.state_history = {}  # Tracks dt and state updates over time
        self.dt_history = []     # Tracks dt history for precision corrections

        self.base_mass = params.get("base_mass", 1.0)
        self.base_moment = params.get("base_moment", 1.0)
        self.speed_of_light = params.get("speed_of_light", 3.0e8)
        self.velocity_threshold = params.get("velocity_threshold", 0.01)
        self.history_length = params.get("history_length", 10)  # Limit history size

    def declare_fields(self):
        """
        Declare fields tracked and updated by this operator instance.
        """
        return self.fields

    def update_state(self, state_tensor, edges, dt):
        """
        Perform discrete integration or derivation for motion states across graph edges.

        Args:
            state_tensor (dict): Dictionary of state tensors for fields (e.g., position, velocity).
            edges (torch.Tensor): Edge indices for graph vertices.
            dt (float): Time step for this update.
        
        Returns:
            Updated state_tensor (dict): State tensor with propagated field updates.
        """
        i, j = edges  # Source and target vertices
        c_squared = self.speed_of_light ** 2

        # Extract states
        pos = state_tensor.get("position", None)
        velocity = state_tensor.get("velocity", None)
        momentum = state_tensor.get("momentum", None)

        if pos is None or velocity is None:
            raise ValueError("Position and velocity fields must be present in the state tensor.")

        # Compute displacements and velocities
        r_ij = pos[j] - pos[i]  # Edge displacements
        v_mag = torch.norm(velocity, dim=1, keepdim=True)  # Velocity magnitudes

        # Relativistic Lorentz gamma factor
        relativistic_mask = (v_mag / self.speed_of_light) > self.velocity_threshold
        gamma = torch.ones_like(v_mag)
        gamma[relativistic_mask] = 1.0 / torch.sqrt(1 - (v_mag[relativistic_mask] ** 2) / c_squared)

        # Update velocity and position
        new_velocity = velocity + (momentum / self.base_mass) * dt if "momentum" in self.fields else velocity
        new_pos = pos + new_velocity * dt

        # Update higher-order fields (e.g., jerk, snap)
        if "acceleration" in self.fields:
            acceleration = (new_velocity - velocity) / dt
            state_tensor["acceleration"] = acceleration

        if "jerk" in self.fields:
            jerk = (acceleration - self.state_history.get("acceleration", acceleration)) / dt
            state_tensor["jerk"] = jerk

        # Track state history
        self.state_history["position"] = new_pos.clone()
        self.state_history["velocity"] = new_velocity.clone()
        self.dt_history.append(dt)

        # Trim history length
        if len(self.dt_history) > self.history_length:
            self.dt_history.pop(0)

        # Update state tensor
        state_tensor["position"] = new_pos
        state_tensor["velocity"] = new_velocity

        return state_tensor

    def reset_history(self):
        """
        Reset state history and dt history for this operator.
        """
        self.state_history = {}
        self.dt_history = []
