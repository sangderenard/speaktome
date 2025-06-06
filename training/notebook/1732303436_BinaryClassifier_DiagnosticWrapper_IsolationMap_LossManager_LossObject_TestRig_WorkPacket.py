import torch
import hashlib
import logging
import time
import yaml
from typing import Dict, Any, List
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class DiagnosticWrapper:
    def __init__(self, name: str, model: torch.nn.Module):
        self.name = name
        self.model = model
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - {self.name} - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Adjust granularity here
        return logger

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        # Log input
        self.logger.debug(f"Input Args: {[(arg.shape if isinstance(arg, torch.Tensor) else type(arg)) for arg in args]}")
        try:
            # Execute wrapped model
            output = self.model(*args, **kwargs)
            # Log output
            self.logger.debug(f"Output: {output}")
            return output
        except Exception as e:
            self.logger.error(f"Exception in {self.name}: {e}")
            raise e

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.model, attr)

class LossManager:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.loss_queue: List[torch.Tensor] = []
        self.teed_tensors: set = set()  # Track teed tensors by hash
        self.submitted_tensors: set = set()
        self.optimizer = optimizer

    def track_tee(self, tensor_hash: str):
        """
        Register a teed tensor that must be accounted for before releasing loss.
        """
        self.teed_tensors.add(tensor_hash)

    def add_loss(self, tensor_hash: str, loss_value: torch.Tensor):
        """
        Submit a tensor for loss calculation, ensuring it is accounted for.
        """
        if tensor_hash not in self.teed_tensors:
            raise ValueError(f"Tensor {tensor_hash} is not registered as part of a tee.")
        self.submitted_tensors.add(tensor_hash)
        self.loss_queue.append(loss_value)
        logging.getLogger("LossManager").debug(f"Loss added for tensor {tensor_hash}: {loss_value.item()}")

    def finalize_loss(self) -> torch.Tensor:
        """
        Compute and release the loss only if all teed tensors are accounted for.
        """
        if not self.teed_tensors.issubset(self.submitted_tensors):
            missing_tensors = self.teed_tensors - self.submitted_tensors
            raise ValueError(f"Missing submissions for teed tensors: {missing_tensors}")
        
        # Compute total loss
        total_loss = sum(self.loss_queue)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        logging.getLogger("LossManager").info(f"Total loss finalized: {total_loss.item()}")

        # Clear state
        self.loss_queue.clear()
        self.teed_tensors.clear()
        self.submitted_tensors.clear()
        return total_loss

class LossObject:
    def __init__(self, value: torch.Tensor, tensor_hash: str, packet: 'WorkPacket', source: str, metadata: Dict[str, Any]):
        self.value = value
        self.tensor_hash = tensor_hash
        self.packet = packet
        self.source = source
        self.metadata = metadata

class IsolationMap:
    def __init__(self):
        self.validation_queue: List[Dict[str, Any]] = []
        self.teed_paths: Dict[str, Dict[str, Any]] = {}
        self.clock = 0  # Current clock cycle

    def queue_tensor_for_validation(self, tensor: torch.Tensor, metadata: Dict[str, Any], step: int):
        """
        Queue tensor for validation with its metadata and clock cycle step.
        """
        self.validation_queue.append({"tensor": tensor, "metadata": metadata, "step": step})
        logging.getLogger("IsolationMap").debug(f"Tensor queued for validation: {metadata['hash']} at step {step}")

    def register_tee(self, tee_id: str, destinations: List[str]):
        """
        Register a tee with its destinations.
        """
        self.teed_paths[tee_id] = {"destinations": destinations, "returned": set()}
        logging.getLogger("IsolationMap").info(f"Tee registered: {tee_id} with destinations {destinations}")

    def validate_tensor(self, current_step: int):
        """
        Validate tensors in the queue for the current clock cycle step.
        """
        for item in self.validation_queue:
            tensor, metadata, step = item["tensor"], item["metadata"], item["step"]
            if step != current_step:
                raise ValueError(f"Tensor scheduled for step {step} presented at step {current_step}.")

            current_hash = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
            if current_hash != metadata["hash"]:
                raise ValueError(f"Validation failed: Hash mismatch for tensor at step {step}.")

            logging.getLogger("IsolationMap").info(f"Tensor validated: {current_hash} at step {step}")
        
        self.validation_queue.clear()
        self.clock += 1

    def validate_tee_return(self, tee_id: str, tensor_name: str) -> bool:
        """
        Mark a teed tensor as returned.
        """
        if tee_id not in self.teed_paths:
            raise KeyError(f"Tee ID {tee_id} not found in registered paths.")
        self.teed_paths[tee_id]["returned"].add(tensor_name)
        logging.getLogger("IsolationMap").debug(f"Tee {tee_id} returned tensor: {tensor_name}")

        # Check if all destinations have returned
        expected = set(self.teed_paths[tee_id]["destinations"])
        returned = self.teed_paths[tee_id]["returned"]
        if returned == expected:
            del self.teed_paths[tee_id]  # Tee fully accounted for
            logging.getLogger("IsolationMap").info(f"Tee {tee_id} fully accounted for and removed.")
            return True
        return False


class WorkPacket:
    def __init__(self, packet_name: str, isolation_map: IsolationMap):
        """
        Initialize the WorkPacket.

        Parameters:
        - packet_name (str): Unique name for the WorkPacket (max 50 chars).
        - isolation_map (IsolationMap): Reference to the Isolation Map for tensor path validation.
        """
        if len(packet_name) > 50:
            raise ValueError("WorkPacket name must not exceed 50 characters.")
        self.packet_name = packet_name
        self.tensors: Dict[str, Dict[str, Any]] = {}  # Store managed tensors by name
        self.logger = self._setup_logger()
        self.history: List[Dict[str, Any]] = []  # Detailed history of actions
        self.isolation_map = isolation_map

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.packet_name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - {self.packet_name} - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    def add_tensor(self, description: str, name: str, tensor: torch.Tensor, rationale: str, notes: str = "") -> str:
        """
        Add a tensor to the WorkPacket with extensive metadata and validation.

        Parameters:
        - description (str): Minimum length of 10 characters, explaining the tensor's purpose.
        - name (str): Unique identifier, max length 50 characters.
        - tensor (torch.Tensor): The tensor to manage.
        - rationale (str): Explanation for why the tensor is being added.
        - notes (str): Optional notes for additional context.
        """
        # Validate inputs
        if len(description) < 25:
            raise ValueError("Description must be at least 25 characters long.")
        if len(name) < 10:
            raise ValueError("Name must exceed 10 characters.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Only torch.Tensor objects can be added.")
        if name in self.tensors:
            print(self.tensors)
            raise ValueError(f"A tensor with name '{name}' already exists in this WorkPacket.")

        # Generate hash and store metadata
        tensor_hash = self._generate_hash(tensor)
        timestamp = self._generate_timestamp_hash()
        metadata = {
            "description": description,
            "rationale": rationale,
            "notes": notes,
            "timestamp": timestamp,
            "hash": tensor_hash,
            "version": tensor._version,
            "state": "created",
        }
        self.tensors[name] = {"tensor": tensor, "metadata": metadata}
        self._log_action("add_tensor", name=name, metadata=metadata)
        
        return tensor_hash

    def _generate_hash(self, tensor: torch.Tensor) -> str:
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()

    def _generate_timestamp_hash(self) -> str:
        return hashlib.sha256(str(time.time()).encode('utf-8')).hexdigest()

    def verify_tensor(self, name: str) -> bool:
        """
        Verify the integrity of a tensor in the WorkPacket.
        """
        if name not in self.tensors:
            raise KeyError(f"No tensor with name '{name}' found in this WorkPacket.")

        tensor_data = self.tensors[name]
        tensor = tensor_data["tensor"]
        metadata = tensor_data["metadata"]

        # Check version consistency
        if tensor._version != metadata["version"]:
            self.logger.error(f"Tensor '{name}' version mismatch: Expected {metadata['version']}, Got {tensor._version}")
            metadata["state"] = "invalid"
            return False

        # Update state and log
        metadata["state"] = "verified"
        self._log_action("verify_tensor", name=name, metadata=metadata)
        return True

    def queue_for_validation(self, name: str, step: int):
        """
        Queue a tensor for validation with the Isolation Map before exiting the WorkPacket.
        """
        if name not in self.tensors:
            raise KeyError(f"No tensor with name '{name}' found in this WorkPacket.")

        tensor_data = self.tensors[name]
        self.isolation_map.queue_tensor_for_validation(tensor_data["tensor"], tensor_data["metadata"], step)
        self._log_action("queue_for_validation", name=name, metadata=tensor_data["metadata"], step=step)

    def validate_on_return(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> bool:
        """
        Validate the tensor upon return to the WorkPacket.
        """
        tensor_hash = self._generate_hash(tensor)
        expected_hash = metadata["hash"]

        if tensor_hash != expected_hash:
            self.logger.error("Validation failed: Tensor hash mismatch on return.")
            raise ValueError("Validation failed: Tensor hash mismatch.")

        self._log_action("validate_on_return", metadata=metadata)
        return True

    def submit_for_loss(self, name: str) -> torch.Tensor:
        """
        Submit a tensor for loss calculation. Submission destroys the tensor and creates a new state.
        """
        if name not in self.tensors:
            raise KeyError(f"No tensor with name '{name}' found in this WorkPacket.")

        tensor_data = self.tensors[name]
        metadata = tensor_data["metadata"]

        if metadata["state"] != "verified":
            raise ValueError(f"Tensor '{name}' must be verified before submission for loss.")

        # Clone, detach, and destroy the original tensor
        tensor = tensor_data["tensor"]
        new_tensor = tensor.clone().detach()
        metadata["state"] = "submitted"
        self._log_action("submit_for_loss", name=name, metadata=metadata)

        # Remove tensor from management
        del self.tensors[name]
        return new_tensor

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieve the history of actions performed on this WorkPacket."""
        return self.history

    def _log_action(self, action: str, **kwargs):
        """Log an action and record it in history."""
        self.logger.info(f"Action: {action}, Details: {kwargs}")
        self.history.append({"action": action, "details": kwargs})

class TestRig:
    def __init__(self, yaml_path: str):
        # Load YAML
        self.yaml_content = self._load_yaml(yaml_path)

        # Validate YAML
        validate_yaml(self.yaml_content)

        # Initialize components
        self.isolation_map = IsolationMap()
        self.work_packet = create_work_packet_from_yaml(self.yaml_content, self.isolation_map)
        self.network = create_network_from_yaml(self.yaml_content)
        self.network = DiagnosticWrapper(self.work_packet.packet_name, self.network)
        self.optimizer = Adam(self.network.parameters(), lr=0.01)
        self.loss_fn = nn.BCELoss()
        self.loss_manager = LossManager(self.optimizer)

        # Generate synthetic data
        self.train_data, self.train_labels = generate_synthetic_data(1000)
        self.test_data, self.test_labels = generate_synthetic_data(100)

    def _load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)

    def execute_itinerary(self):
        """Execute the YAML-defined itinerary."""
        tensor_name = f"{self.work_packet.packet_name}"
        for step in self.yaml_content["itinerary"]["steps"]:
            if step["action"] == "initialize":
                # Initialize tensor as per YAML
                self._initialize_tensor(step, tensor_name)
            elif step["action"] == "LayerTraversal":
                # Traverse through a layer
                self._traverse_layer(step, tensor_name)
            elif step["action"] == "submission":
                # Submit tensor for loss
                self._submit_tensor_for_loss(step, tensor_name)

        # Finalize loss
        total_loss = self.loss_manager.finalize_loss()
        print(f"Final Loss: {total_loss.item():.4f}")

    def _initialize_tensor(self, step: Dict[str, Any], tensor_name: str):
        """Initialize the tensor and add to the WorkPacket."""
        tensor = self.train_data
        tensor_hash = self.work_packet.add_tensor(
            description=step["description"],
            name=tensor_name,
            tensor=tensor,
            rationale="Initialize input tensor.",
            notes=step.get("notes", ""),
        )
        print(f"Initialized Tensor: {tensor_hash}")

    def _traverse_layer(self, step: Dict[str, Any], tensor_name: str):
        """Pass tensor through a specified network layer."""
        self.work_packet.queue_for_validation(name=tensor_name, step=step["step"])
        self.isolation_map.validate_tensor(current_step=step["step"])

        # Retrieve tensor
        tensor_data = self.work_packet.tensors[tensor_name]
        tensor = tensor_data["tensor"]

        # Pass through the layer
        predictions = self.network(tensor)
        self.work_packet.add_tensor(
            description=step["description"],
            name=tensor_name,
            tensor=predictions,
            rationale=f"Processed through {step['layer']}.",
            notes=step.get("notes", ""),
        )
        print(f"Processed Tensor: {tensor_name} through {step['layer']}")

    def _submit_tensor_for_loss(self, step: Dict[str, Any], tensor_name: str):
        """Submit tensor for loss calculation."""
        tensor_data = self.work_packet.tensors[tensor_name]
        predictions = tensor_data["tensor"]
        loss = self.loss_fn(predictions, self.train_labels)

        # Add loss to the LossManager
        self.loss_manager.add_loss(tensor_data["metadata"]["hash"], loss)
        print(f"Loss submitted: {loss.item():.4f}")

    def test_model(self):
        """Evaluate the model on test data."""
        with torch.no_grad():
            test_preds = self.network(self.test_data)
            accuracy = ((test_preds > 0.5) == self.test_labels).float().mean().item()
            print(f"Test Accuracy: {accuracy:.2%}")


def validate_yaml(yaml_config: Dict[str, Any]) -> bool:
    """
    Validate the YAML configuration for the WorkPacket.

    Raises:
        ValueError: If any required field is missing or inconsistent.
    """
    # Check required sections
    if "work_packet" not in yaml_config:
        raise ValueError("YAML must include a 'work_packet' section.")
    if "itinerary" not in yaml_config or "steps" not in yaml_config["itinerary"]:
        raise ValueError("Itinerary with steps is required.")
    if "network" not in yaml_config or "layers" not in yaml_config["network"]:
        raise ValueError("Network definition with layers is required.")
    
    # Validate itinerary and network alignment
    itinerary_steps = [step for step in yaml_config["itinerary"]["steps"] if step["action"] == "LayerTraversal"]
    layer_names = [layer["name"] for layer in yaml_config["network"]["layers"]]
    
    for step in itinerary_steps:
        if step["layer"] not in layer_names:
            raise ValueError(f"LayerTraversal step references undefined layer: {step['layer']}.")
    
    # Validate custom operations
    for step in yaml_config["itinerary"]["steps"]:
        if step["action"] == "user_defined" and "cost" not in step:
            raise ValueError(f"Custom operation step '{step['destination']}' must specify a 'cost'.")
    
    # Ensure clock cycles match the number of steps
    expected_clock_cycles = len(yaml_config["itinerary"]["steps"])
    if yaml_config["itinerary"]["clock_cycle"] < expected_clock_cycles:
        raise ValueError("Clock cycle count is less than the number of itinerary steps.")
    
    return True

def create_work_packet_from_yaml(yaml_config: Dict[str, Any], isolation_map: IsolationMap) -> WorkPacket:
    """
    Create WorkPacket directly from the already-parsed YAML dictionary.
    """
    validate_yaml(yaml_config)

    packet_name = yaml_config["work_packet"]["name"]
    wp = WorkPacket(packet_name, isolation_map)

    for step in yaml_config["itinerary"]["steps"]:
        if step["action"] == "initialize":
            if yaml_config["initialization"]["method"] == "rand":
                tensor = torch.randn(128, 128)  # Example initialization
            elif yaml_config["initialization"]["method"] == "ones":
                tensor = torch.ones(128, 128)
            elif yaml_config["initialization"]["method"] == "zeros":
                tensor = torch.zeros(128, 128)
            elif yaml_config["initialization"]["method"] == "texture":
                # Load tensor from texture path
                tensor = torch.randn(128, 128)  # Placeholder for texture logic
            else:
                raise ValueError(f"Unknown initialization method: {yaml_config['initialization']['method']}")

            wp.add_tensor(
                description=step["description"],
                name=f"{packet_name}",
                tensor=tensor,
                rationale="System initialization",
                notes=step.get("notes", ""),
            )
        elif step["action"] == "LayerTraversal":
            layer_name = step["layer"]
            wp.queue_for_validation(name=f"{packet_name}", step=step["step"])
        elif step["action"] == "tee":
            tee_id = f"tee_{step['step']}"
            destinations = [dest["destination"] for dest in step["destinations"]]
            isolation_map.register_tee(tee_id, destinations)

    return wp




def load_yaml_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def create_network_from_yaml(yaml_config):
    network_layers = yaml_config["network"]["layers"]
    for layer in network_layers:
        if layer["name"] == "Classifier":
            params = layer["parameters"]
            return BinaryClassifier(
                input_size=params["input_size"],
                output_size=params["output_size"]
            )
    raise ValueError("No valid network configuration found.")


def generate_synthetic_data(num_samples=1000):
    """
    Generate a synthetic binary classification dataset.
    """
    X = torch.rand(num_samples, 2) * 2 - 1  # Random data in range [-1, 1]
    y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)  # Label: 1 if product > 0, else 0
    return X, y

def main():
    # Path to the YAML configuration
    yaml_path = "example.yaml"

    # Create and execute the test rig
    rig = TestRig(yaml_path)
    rig.execute_itinerary()
    rig.test_model()


if __name__ == "__main__":
    main()
