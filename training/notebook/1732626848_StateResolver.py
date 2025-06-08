import torch
from typing import Optional, List, Dict
import copy
import logging
from state import State

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class StateResolver:
    """
    Resolves sparse or incomplete states by filling missing values, assigning confidence weights,
    and validating resolved states.
    """

    def __init__(self, default_fill: float = 0.0):
        """
        Initializes the StateResolver with default values for missing subfeatures.

        Args:
            default_fill (float): Default value to fill for missing subfeatures.
        """
        self.default_fill = default_fill

    def resolve(self, state: "State", previous_state: Optional["State"] = None, stream: Optional[List["State"]] = None) -> Dict:
        """
        Resolves a sparse or incomplete state into a fully populated state.

        Args:
            state (State): The current state to resolve.
            previous_state (Optional[State]): The previous state for interpolation (if available).
            stream (Optional[List[State]]): A stream of historical states for temporal context.

        Returns:
            Dict: A dictionary containing:
                - "resolved_state": Fully populated State.
                - "confidence_weights": Confidence map for the resolved state.
        """
        # Start with a deep copy of the current state to avoid modifying the original
        resolved_state = copy.deepcopy(state)

        # Initialize confidence weights (same shape as subfeatures, initially all zeros)
        confidence_weights = {
            feature: torch.zeros(len(state.SUBFEATURE_MAP)) for feature in state.state["features"]
        }

        # Resolve each feature
        for feature, feature_data in state.state["features"].items():
            resolved_subfeatures = []
            confidences = []
            for i, value in enumerate(feature_data["subfeatures"]):
                if value is not None:
                    # Use provided value
                    resolved_subfeatures.append(value)
                    confidences.append(1.0)  # Full confidence for provided values
                elif previous_state and isinstance(previous_state, State):
                    # Interpolate if previous state is available
                    # Extract dt from the current state or default to 1.0 if not available
                    dt = state.state.get("time", {}).get("dt", 1.0)

                    interpolated_value, _ = self.interpolate(
                        previous_state.state["features"][feature]["subfeatures"][i],
                        value,
                        dt,  # Pass the extracted dt
                    )

                    resolved_subfeatures.append(interpolated_value)
                    confidences.append(0.0)  # Interpolated values have zero confidence
                else:
                    # Fill with default value if no previous state
                    resolved_subfeatures.append(self.default_fill)
                    confidences.append(0.0)  # Filled values have zero confidence

            # Update resolved state and confidence weights
            resolved_state.state["features"][feature]["subfeatures"] = resolved_subfeatures
            confidence_weights[feature] = torch.tensor(confidences)

        # Assign confidence weights using the corrected method
        confidence_weights = self.assign_confidence_weights(resolved_state, previous_state)

        # Validate resolved state if needed
        if not self.validate(resolved_state):
            raise ValueError("Resolved state failed validation.")

        return {
            "resolved_state": resolved_state,
            "confidence_weights": confidence_weights,
        }

    def interpolate(self, previous_value: Optional[float], current_value: Optional[float], dt: float) -> (float, float):
        """
        Interpolates between a previous and current value based on the time delta.

        Args:
            previous_value (Optional[float]): The previous value.
            current_value (Optional[float]): The current value.
            dt (float): The time delta for interpolation.

        Returns:
            Tuple[float, float]: The interpolated value and the interpolation degree (0 to 1).
        """
        if previous_value is not None and current_value is not None:
            interpolated_value = (previous_value + current_value) / 2  # Simple average
            interpolation_degree = 1.0  # Full interpolation
        elif previous_value is not None:
            interpolated_value = previous_value
            interpolation_degree = 0.5  # Partial interpolation
        elif current_value is not None:
            interpolated_value = current_value
            interpolation_degree = 0.5  # Partial interpolation
        else:
            interpolated_value = self.default_fill
            interpolation_degree = 0.0  # No interpolation

        logger.debug(f"Interpolated value: {interpolated_value} with degree {interpolation_degree}")
        return interpolated_value, interpolation_degree

    def assign_confidence_weights(self, resolved_state: "State", previous_state: Optional["State"] = None) -> Dict[str, torch.Tensor]:
        """
        Assigns confidence weights to a resolved state based on authority and interpolation degree.

        Args:
            resolved_state (State): The resolved state.
            previous_state (Optional[State]): The previous state for interpolation reference.

        Returns:
            Dict[str, torch.Tensor]: Updated confidence weights.
        """
        confidence_weights = {
            feature: torch.zeros(len(resolved_state.SUBFEATURE_MAP)) for feature in resolved_state.state["features"]
        }

        for feature_index, (feature, feature_data) in enumerate(resolved_state.state["features"].items()):
            for i, value in enumerate(feature_data["subfeatures"]):
                if value is None or value == self.default_fill:
                    # No confidence for default or missing values
                    confidence_weights[feature][i] = 0.0
                else:
                    # Scale confidence based on authority and interpolation degree
                    authority_curr = resolved_state.state["authority_mask"][feature_index, i].item()
                    if previous_state and isinstance(previous_state, State):
                        prev_value = previous_state.state["features"][feature]["subfeatures"][i]
                        if prev_value is None or prev_value == self.default_fill:
                            confidence_interpolation = 0.5  # Lower confidence for sparse history
                        else:
                            # Interpolation degree: similarity between current and previous
                            diff = abs(value - prev_value) if isinstance(value, (float, int)) else 0.0
                            confidence_interpolation = max(1.0 - diff, 0.0)  # Confidence decreases with greater difference
                    else:
                        confidence_interpolation = 1.0  # Full confidence without previous state

                    # Combine authority and interpolation confidence
                    confidence_weights[feature][i] = authority_curr * confidence_interpolation

        return confidence_weights

    def validate(self, state: "State") -> bool:
        """
        Validates a resolved state.

        Args:
            state (State): The resolved state.

        Returns:
            bool: True if the state is valid, False otherwise.
        """
        return state.validate_state()

    def batch_resolve(self, states: List["State"]) -> List[Dict]:
        """
        Resolves a batch of states.

        Args:
            states (List[State]): A list of states to resolve.

        Returns:
            List[Dict]: A list of dictionaries containing resolved states and their confidence weights.
        """
        resolved_batch = []
        previous_state = None
        for state in states:
            resolved = self.resolve(state, previous_state)
            resolved_batch.append(resolved)
            previous_state = state  # Update previous state for next iteration
        return resolved_batch
def test_state_resolver():
    """
    Thoroughly tests the State and StateResolver classes.
    """
    # Initialize StateResolver
    resolver = StateResolver(default_fill=0.0)

    # Create initial state
    initial_state = State(object_id="obj_001")
    initial_state.state["features"]["position"]["subfeatures"][0] = 1.0  # x
    initial_state.state["features"]["position"]["subfeatures"][1] = 2.0  # y
    initial_state.state["features"]["position"]["subfeatures"][2] = 3.0  # z
    initial_state.state["features"]["velocity"]["subfeatures"][0] = 0.1  # vx
    initial_state.state["features"]["velocity"]["subfeatures"][1] = 0.2  # vy
    initial_state.state["features"]["velocity"]["subfeatures"][2] = 0.3  # vz
    initial_state.state["features"]["acceleration"]["subfeatures"][0] = 0.01  # ax
    initial_state.state["features"]["acceleration"]["subfeatures"][1] = 0.02  # ay
    initial_state.state["features"]["acceleration"]["subfeatures"][2] = 0.03  # az
    initial_state.state["features"]["jerk"]["subfeatures"][0] = 0.001  # jx
    initial_state.state["features"]["jerk"]["subfeatures"][1] = 0.002  # jy
    initial_state.state["features"]["jerk"]["subfeatures"][2] = 0.003  # jz
    initial_state.state["features"]["temperature"]["subfeatures"][7] = 300.0  # Temperature

    # Set temporal dynamics
    initial_state.state["time"]["sim_time"] = 0.0
    initial_state.state["time"]["dt"] = 0.1
    initial_state.state["time"]["time_in_dt"] = None

    # Validate initial state
    assert initial_state.validate_state(), "Initial state should be valid."

    logger.info("Initial State:")
    print(initial_state)

    # Create new state with some missing subfeatures
    new_state = State(object_id="obj_001")
    new_state.state["features"]["position"]["subfeatures"][0] = 1.5  # x updated
    # y is missing
    new_state.state["features"]["position"]["subfeatures"][2] = 3.5  # z updated
    new_state.state["features"]["velocity"]["subfeatures"][0] = 0.15  # vx updated
    # vy and vz are missing
    new_state.state["features"]["acceleration"]["subfeatures"][0] = 0.015  # ax updated
    # ay and az are missing
    # Jerk subfeatures are missing
    new_state.state["features"]["temperature"]["subfeatures"][7] = 305.0  # Temperature updated

    # Set temporal dynamics
    new_state.state["time"]["sim_time"] = 0.1
    new_state.state["time"]["dt"] = 0.1
    new_state.state["time"]["time_in_dt"] = None

    # Validate new state (should fail since jerk is required but missing)
    assert not new_state.validate_state(), "New state should be invalid due to missing jerk subfeatures."

    # Resolve new state using the resolver and initial state
    resolved = resolver.resolve(new_state, previous_state=initial_state)

    resolved_state = resolved["resolved_state"]
    confidence_weights = resolved["confidence_weights"]

    logger.info("Resolved State:")
    print(resolved_state)

    logger.info("Confidence Weights:")
    for feature, weights in confidence_weights.items():
        print(f"{feature}: {weights.tolist()}")

    # Validate resolved state
    assert resolved_state.validate_state(), "Resolved state should be valid after interpolation."

    # Demonstrate batch processing
    # Create a stream of states
    state_stream = [initial_state]
    for t in range(1, 6):
        new_state = State(object_id="obj_001")
        new_state.state["features"]["position"]["subfeatures"][0] = 1.0 + 0.5 * t  # x increases
        new_state.state["features"]["position"]["subfeatures"][1] = 2.0 + 0.5 * t  # y increases
        new_state.state["features"]["position"]["subfeatures"][2] = 3.0 + 0.5 * t  # z increases
        new_state.state["features"]["velocity"]["subfeatures"][0] = 0.1 + 0.05 * t  # vx increases
        new_state.state["features"]["velocity"]["subfeatures"][1] = 0.2 + 0.05 * t  # vy increases
        new_state.state["features"]["velocity"]["subfeatures"][2] = 0.3 + 0.05 * t  # vz increases
        new_state.state["features"]["acceleration"]["subfeatures"][0] = 0.01 + 0.005 * t  # ax increases
        new_state.state["features"]["acceleration"]["subfeatures"][1] = 0.02 + 0.005 * t  # ay increases
        new_state.state["features"]["acceleration"]["subfeatures"][2] = 0.03 + 0.005 * t  # az increases
        new_state.state["features"]["jerk"]["subfeatures"][0] = 0.001 + 0.0005 * t  # jx increases
        new_state.state["features"]["jerk"]["subfeatures"][1] = 0.002 + 0.0005 * t  # jy increases
        new_state.state["features"]["jerk"]["subfeatures"][2] = 0.003 + 0.0005 * t  # jz increases
        new_state.state["features"]["temperature"]["subfeatures"][7] = 300.0 + 5 * t  # Temperature increases

        # Set temporal dynamics
        new_state.state["time"]["sim_time"] = t * 0.1
        new_state.state["time"]["dt"] = 0.1
        new_state.state["time"]["time_in_dt"] = None

        # Add to stream
        state_stream.append(new_state)

    # Perform batch resolution
    resolved_batch = resolver.batch_resolve(state_stream)

    logger.info("Batch Resolved States and Confidence Weights:")
    for idx, resolved in enumerate(resolved_batch):
        logger.info(f"State {idx}:")
        print(resolved["resolved_state"])
        print("Confidence Weights:")
        for feature, weights in resolved["confidence_weights"].items():
            print(f"{feature}: {weights.tolist()}")
        print("-" * 50)

    logger.info("All tests passed successfully.")


test_state_resolver()