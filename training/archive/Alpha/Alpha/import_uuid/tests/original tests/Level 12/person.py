class NodeBrain:
    def __init__(self, node_id):
        self.node_id = node_id
        self.targets = {}  # Key: target_id, Value: Reference object
        self.best_losses = []  # Stores best loss metadata
        self.local_weights = {}  # Stores local NN weights
        self.edge_weights = {}  # Personal edge weights for the GNN
        self.active_tasks = {}  # Current tasks for targets
        self.history = []  # Log of completed tasks

    def add_target(self, reference):
        """Add a new target for the node to manage."""
        self.targets[reference.id] = reference

    def remove_target(self, reference_id):
        """Remove a target from the node's management."""
        if reference_id in self.targets:
            del self.targets[reference_id]

    def update_best_losses(self, loss, output, target_id):
        """Update best loss history for a specific target."""
        self.best_losses.append({
            "loss": loss,
            "output": output.detach().cpu(),
            "timestamp": time.time(),
            "target_id": target_id
        })
        self.best_losses = sorted(self.best_losses, key=lambda x: x["loss"])[:BUFFER_SIZE]  # Keep top results

    def snapshot_weights(self, generator, discriminator, edges):
        """Capture local and edge weights."""
        self.local_weights = {
            "generator": {name: param.clone() for name, param in generator.named_parameters()},
            "discriminator": {name: param.clone() for name, param in discriminator.named_parameters()}
        }
        self.edge_weights = {edge.id: edge.weight for edge in edges}

    def log_task(self, task_id, result):
        """Log the completion of a task."""
        self.history.append({"task_id": task_id, "result": result, "timestamp": time.time()})

    def generate_report(self):
        """Generate a self-report of the node's current state."""
        return {
            "node_id": self.node_id,
            "targets": list(self.targets.keys()),
            "best_losses": self.best_losses,
            "local_weights": list(self.local_weights.keys()),
            "edge_weights": self.edge_weights,
            "active_tasks": self.active_tasks
        }
class Node:
    def __init__(self, features=None):
        self.id = self._generate_id()
        self.features = features if features is not None else torch.empty(0)
        self.edges = []
        self.task_completed = False

        # Neural Networks
        self.generator = self._create_generator()
        self.discriminator = self._create_discriminator()

        # Optimizers
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        # Loss functions
        self.gen_loss_fn = nn.MSELoss()
        self.disc_loss_fn = nn.MSELoss()

        # Delegate brain for organizational logic
        self.brain = NodeBrain(node_id=self.id)

    def _generate_id(self):
        return uuid.uuid4().hex

    def _create_generator(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _create_discriminator(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def process_task(self, input_tensor, target_id, focus="generator"):
        reference = self.brain.targets.get(target_id)
        if not reference:
            raise ValueError(f"Target {target_id} not found for Node {self.id}")

        if focus == "generator":
            output = self.generator(input_tensor)
            loss = self.gen_loss_fn(output, reference.reference_tensor)
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()

            # Update brain with task result
            self.brain.update_best_losses(loss.item(), output, target_id)
            return output, loss.item()

        elif focus == "discriminator":
            output = self.discriminator(input_tensor)
            loss = self.disc_loss_fn(output, reference.reference_tensor)
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
            return output, loss.item()

    def snapshot_state(self):
        """Snapshot the current state into the NodeBrain."""
        self.brain.snapshot_weights(self.generator, self.discriminator, self.edges)
class Graph:
    def execute(self, input_tensor):
        for node in self.nodes.values():
            for target_id in node.brain.targets.keys():
                gen_output, gen_loss = node.process_task(input_tensor, target_id, focus="generator")
                disc_output, disc_loss = node.process_task(input_tensor, target_id, focus="discriminator")

                # Determine the focus based on higher loss
                if gen_loss > disc_loss:
                    node.process_task(input_tensor, target_id, focus="generator")
                else:
                    node.process_task(input_tensor, target_id, focus="discriminator")

    def collect_reports(self):
        return [node.brain.generate_report() for node in self.nodes.values()]
