import torch
from laplace import GridDomain, BuildLaplace3D, RectangularTransform

class FieldRegistry:
    """A registry for standard and custom field functions."""
    def __init__(self):
        self.fields = {"density": None, "tension": None}  # Core fields
        self.custom_fields = {}

    def register_field(self, name, function):
        """Add a custom field to the registry."""
        if name in self.fields or name in self.custom_fields:
            raise ValueError(f"Field '{name}' already exists.")
        self.custom_fields[name] = function
        print(f"Registered custom field '{name}'.")

    def get_field(self, name):
        """Retrieve a field function by name."""
        return self.fields.get(name) or self.custom_fields.get(name)

    def list_fields(self):
        """List all registered fields."""
        return list(self.fields.keys()) + list(self.custom_fields.keys())

class FieldComposer:
    """Compose multiple fields with weighted contributions."""
    def __init__(self, registry):
        self.registry = registry
        self.composed_fields = []

    def add_field(self, name, weight=1.0):
        """Add a weighted field to the composer."""
        field = self.registry.get_field(name)
        if field is None:
            raise ValueError(f"Field '{name}' not found.")
        self.composed_fields.append((field, weight))
        print(f"Added field '{name}' with weight {weight}.")

    def evaluate(self, x, y, z):
        """Evaluate the composed fields at a given point."""
        total = 0
        for field, weight in self.composed_fields:
            total += weight * field(x, y, z)
        return total

class MetricModifier:
    """Dynamically modify metric tensors."""
    def __init__(self):
        self.modifiers = []

    def add_modifier(self, name, function):
        """Register a metric modification function."""
        self.modifiers.append((name, function))
        print(f"Registered metric modifier '{name}'.")

    def apply_modifiers(self, metric_tensor, *args):
        """Apply all registered modifiers to a given metric tensor."""
        for name, modifier in self.modifiers:
            print(f"Applying '{name}' modifier...")
            metric_tensor = modifier(metric_tensor, *args)
        return metric_tensor
def tensor_clensor(x):
    return torch.tensor(x)
class LaplaceHelper:
    """Main bridge for 3D Laplace simulation management."""
    def __init__(self):
        self.registry = FieldRegistry()
        self.composer = FieldComposer(self.registry)
        self.metric_modifier = MetricModifier()
        self.metric_tensor = torch.zeros((3, 3), dtype=torch.float64)  # Default 3D metric tensor

        # Pre-register standard fields
        self.registry.fields["density"] = lambda x, y, z: torch.sin(tensor_clensor(x)) + torch.cos(tensor_clensor(y))
        self.registry.fields["tension"] = lambda x, y, z: torch.exp(-((tensor_clensor(x)**2 + tensor_clensor(y)**2 + tensor_clensor(z)**2)))

    def setup_fields(self):
        """Interactive or scripted setup for fields and weights."""
        print("Welcome to Laplace Field Setup!")
        print("Available Fields:", self.registry.list_fields())

        # Example: Default field composition (can be user-customized)
        self.composer.add_field("density", weight=1.5)
        self.composer.add_field("tension", weight=0.5)

        # Example: Custom fields can be registered here
        self.registry.register_field("custom_pressure", lambda x, y, z: x * y * z)

    def add_metric_modifier(self, name, function):
        """Add a metric modification function."""
        self.metric_modifier.add_modifier(name, function)

    def evaluate_fields(self, x, y, z):
        """Evaluate the field composition at a given point."""
        result = self.composer.evaluate(x, y, z)
        print(f"Field Evaluation at ({x}, {y}, {z}): {result}")
        return result

    def modify_metric(self, *args):
        """Apply modifiers to the metric tensor."""
        self.metric_tensor = self.metric_modifier.apply_modifiers(self.metric_tensor, *args)
        print("Modified Metric Tensor:\n", self.metric_tensor)
        return self.metric_tensor

    def run(self, x, y, z):
        """Run a complete pipeline: evaluate fields and apply modifiers."""
        print("\n--- Running Laplace Helper Pipeline ---")
        field_value = self.evaluate_fields(x, y, z)
        modified_metric = self.modify_metric(x, y, z)
        return field_value, modified_metric

# Example Usage
if __name__ == "__main__":
    laplace = LaplaceHelper()
    laplace.setup_fields()

    # Add a custom metric modifier
    laplace.add_metric_modifier(
        "scale_adjustment",
        lambda tensor, x, y, z: tensor + torch.tensor([[x, y, z], [y, x, z], [z, y, x]])
    )

    # Run pipeline for a sample point
    laplace.run(1.0, 2.0, 3.0)
