import torch
from typing import Dict, Any, Optional, List
import copy
import logging
from scipy.sparse import coo_matrix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class State:
    """
    Represents the physical state of an object, supporting structured subfeatures
    with a formalized hierarchy for positional, rotational, and other critical attributes.
    """

    def __init__(self, object_id: str):
        """
        Initializes the State with structured features and subfeatures.

        Args:
            object_id (str): The unique identifier for this state.
        """
        # Define formal subfeature map
        self.SUBFEATURE_MAP = [
            "x", "y", "z",        # Cartesian coordinates
            "q1", "q2", "q3", "q4",  # Quaternion components
            "r",                  # Radius (spherical coordinate)
            "p",                  # Phi (azimuthal angle)
            "theta",              # Theta (polar angle)
            "temperature"         # Temperature
        ]

        # Create a structured state object
        self.state = {
            "id": object_id,
            "type": "physical",
            "features": {                           # Features as hierarchical headings
                "position": {"subfeatures": [None] * len(self.SUBFEATURE_MAP)},
                "velocity": {"subfeatures": [None] * len(self.SUBFEATURE_MAP)},
                "acceleration": {"subfeatures": [None] * len(self.SUBFEATURE_MAP)},
                "jerk": {"subfeatures": [None] * len(self.SUBFEATURE_MAP)},
                "snap": {"subfeatures": [None] * len(self.SUBFEATURE_MAP)},
                "crackle": {"subfeatures": [None] * len(self.SUBFEATURE_MAP)},
                "pop": {"subfeatures": [None] * len(self.SUBFEATURE_MAP)},
                "temperature": {"subfeatures": [None] * len(self.SUBFEATURE_MAP)}
            },
            "authority_mask": torch.ones((8, len(self.SUBFEATURE_MAP))),  # Feature x Subfeature
            "time": {                               # Temporal dynamics
                "sim_time": None,                   # Current simulation time
                "dt": None,                         # Time delta for each simulation step
                "time_in_dt": None,                 # Optional event timing within dt
            },
            "mask": {                               # Validation mask for required features
                "position": True,
                "velocity": True,
                "acceleration": True,
                "jerk": True,
                "snap": False,
                "crackle": False,
                "pop": False,
                "temperature": True
            },
            "validator": self.validate_state,       # Reference to the validator function
        }

        # Previous and current states for interpolation
        self.previous_state: Optional[State] = None
        self.current_state: Optional[State] = None
        self.interpolated_state: Optional[State] = None

        logger.info(f"State initialized for object ID: {object_id}")

    def validate_state(self) -> bool:
        """
        Validates that required subfeatures are present for each feature based on the mask.

        Returns:
            bool: True if the state is valid, False otherwise.
        """
        for feature, required in self.state["mask"].items():
            if required:
                subfeatures = self.state["features"][feature]["subfeatures"]
                if all(val is None for val in subfeatures):
                    logger.warning(f"Validation failed: All subfeatures are missing for '{feature}'.")
                    return False
        logger.info("State validation passed.")
        return True

    def update(self, new_state: "State") -> "State":
        """
        Updates the current state by incorporating a new state and interpolating
        the necessary intermediate state based on the authority map.

        Args:
            new_state (State): The new state to be incorporated.

        Returns:
            State: A composite state containing the previous, current, and interpolated states.
        """
        if not isinstance(new_state, State):
            raise ValueError("new_state must be an instance of State.")

        # Preserve the current state as the "previous" state
        self.previous_state = copy.deepcopy(self)

        # Set the new state as the "current" state
        self.current_state = copy.deepcopy(new_state)

        # Interpolate between previous and current states
        self.interpolated_state = self.interpolate(self.previous_state, self.current_state)

        return self.interpolated_state

    def interpolate(self, previous_state: Optional["State"], current_state: "State") -> "State":
        """
        Creates an interpolated state between the previous and current states, weighted by the 2D authority mask.

        Args:
            previous_state (State): The previous state.
            current_state (State): The current state.

        Returns:
            State: An interpolated state.
        """
        interpolated_state = State(self.state["id"])

        for feature_index, (feature, feature_data) in enumerate(current_state.state["features"].items()):
            prev_subfeatures = (
                previous_state.state["features"][feature]["subfeatures"]
                if previous_state
                else [None] * len(feature_data["subfeatures"])
            )
            curr_subfeatures = feature_data["subfeatures"]

            # Use the 2D authority mask to weigh interpolated values
            authority_prev = previous_state.state["authority_mask"][feature_index] if previous_state else torch.zeros(len(curr_subfeatures))
            authority_curr = current_state.state["authority_mask"][feature_index]

            interpolated_subfeatures = [
                self._weighted_combine(prev, curr, a_prev, a_curr)
                for prev, curr, a_prev, a_curr in zip(
                    prev_subfeatures, curr_subfeatures, authority_prev, authority_curr
                )
            ]

            interpolated_state.state["features"][feature]["subfeatures"] = interpolated_subfeatures
            # Update authority mask based on interpolation degree (example: average authority)
            interpolated_state.state["authority_mask"][feature_index] = (authority_prev + authority_curr) / 2

        # Copy temporal dynamics
        interpolated_state.state["time"] = copy.deepcopy(current_state.state["time"])

        # Copy mask
        interpolated_state.state["mask"] = copy.deepcopy(current_state.state["mask"])

        # Validate the interpolated state
        if not interpolated_state.validate_state():
            logger.warning("Interpolated state failed validation.")
        else:
            logger.info("Interpolated state is valid.")

        return interpolated_state

    def _weighted_combine(self, prev: Optional[float], curr: Optional[float], a_prev: float, a_curr: float) -> Optional[float]:
        """
        Combines previous and current subfeature values weighted by their authority.

        Args:
            prev (Optional[float]): Previous subfeature value.
            curr (Optional[float]): Current subfeature value.
            a_prev (float): Authority of the previous value.
            a_curr (float): Authority of the current value.

        Returns:
            Optional[float]: Combined value based on authority weighting.
        """
        if prev is None and curr is None:
            return None
        elif prev is None:
            return curr
        elif curr is None:
            return prev
        else:
            total_authority = a_prev + a_curr
            if total_authority > 0:
                return (prev * a_prev + curr * a_curr) / total_authority
            else:
                return curr

    def make_sparse(self) -> Dict[str, coo_matrix]:
        """
        Converts the state's subfeature data into a sparse COO matrix for storage efficiency.

        Returns:
            Dict[str, coo_matrix]: A dictionary where keys are feature names and values are sparse COO matrices.
        """
        sparse_data = {}
        for feature, feature_data in self.state["features"].items():
            subfeatures = feature_data["subfeatures"]
            indices = [i for i, value in enumerate(subfeatures) if value is not None]
            values = [subfeatures[i] for i in indices]
            if indices:
                sparse_data[feature] = coo_matrix((values, ([0] * len(indices), indices)), shape=(1, len(subfeatures)))
            else:
                sparse_data[feature] = coo_matrix((1, len(subfeatures)))
        return sparse_data

    def make_dense(self, sparse_data: Dict[str, coo_matrix]) -> None:
        """
        Converts the sparse COO matrices back into the state's subfeature data.

        Args:
            sparse_data (Dict[str, coo_matrix]): A dictionary where keys are feature names and values are sparse COO matrices.
        """
        for feature, sparse_matrix in sparse_data.items():
            dense_array = sparse_matrix.toarray().flatten()
            self.state["features"][feature]["subfeatures"] = [
                dense_array[i] if dense_array[i] != 0 else None for i in range(len(dense_array))
            ]

    def __str__(self):
        """
        Provides a string representation of the current state, including interpolation details.

        Returns:
            str: A human-readable summary of the state.
        """
        features_str = "\n".join(
            f"{feature}: {data['subfeatures']}"
            for feature, data in self.state["features"].items()
        )
        return f"State(ID: {self.state['id']})\nFeatures:\n{features_str}\nSubfeature Map: {self.SUBFEATURE_MAP}"

