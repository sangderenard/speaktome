import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
# Helper Function for Network Isolation

# Rolling Cache Class
class RollingCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = []

    def add(self, item):
        if len(self.cache) >= self.max_size:
            self._remove_most_similar_items()
        self.cache.append(item)

    def _remove_most_similar_items(self):
        if len(self.cache) < 2:
            self.cache.pop()
            return
        # Calculate pairwise similarity using Euclidean distance
        data = torch.stack(self.cache)
        data_np = data.view(len(self.cache), -1).cpu().numpy()
        distances = cdist(data_np, data_np, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        # Remove the two most similar items
        indices_to_remove = sorted([i, j], reverse=True)
        for idx in indices_to_remove:
            del self.cache[idx]

    def get_all(self):
        return self.cache

# Small Convolutional Network to Predict Gaussian Parameters
class ConvGaussianPredictor(nn.Module):
    def __init__(self):
        super(ConvGaussianPredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc_mu = nn.Linear(64, 1)  # Predicts mean
        self.fc_sigma = nn.Linear(64, 1)  # Predicts standard deviation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x))  # Ensure sigma is positive
        return mu, sigma

# Binary Classifier Network
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Gaussian Reproducer Network
class GaussianReproducer(nn.Module):
    def __init__(self):
        super(GaussianReproducer, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        return x

# Custom Dataset (Synthetic Gaussian Textures)
class SyntheticGaussianDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=32):
        self.num_samples = num_samples
        self.image_size = image_size
        self.data, self.params = self._generate_data()

    def _generate_data(self):
        data = []
        params = []
        for _ in range(self.num_samples):
            mu = np.random.uniform(-1, 1, size=(1,))
            sigma = np.random.uniform(0.5, 1.5, size=(1,))
            grid = np.linspace(-3, 3, self.image_size)
            x, y = np.meshgrid(grid, grid)
            z = np.exp(-((x - mu[0])**2 + (y - mu[0])**2) / (2 * sigma[0]**2))
            z = z / z.max()
            data.append(z.astype(np.float32))
            params.append((mu[0], sigma[0]))
        data = np.expand_dims(np.array(data), axis=1)  # Shape: (N, 1, H, W)
        params = np.array(params, dtype=np.float32)    # Shape: (N, 2)
        return torch.tensor(data), torch.tensor(params)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.params[idx]

# Instantiate Models
conv_gaussian_predictor = ConvGaussianPredictor()
binary_classifier = BinaryClassifier()
gaussian_reproducer = GaussianReproducer()
discriminator = Discriminator()

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv_gaussian_predictor.to(device)
binary_classifier.to(device)
gaussian_reproducer.to(device)
discriminator.to(device)

# Optimizers
optimizer_predictor = optim.Adam(conv_gaussian_predictor.parameters(), lr=0.001)
optimizer_classifier = optim.Adam(binary_classifier.parameters(), lr=0.001)
optimizer_reproducer = optim.Adam(gaussian_reproducer.parameters(), lr=0.001)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)

# Loss Functions
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCELoss()

# Rolling Cache
rolling_cache = RollingCache(max_size=10)

# Training Step
def train_step(input_texture, true_mu, true_sigma):
    input_texture = input_texture.to(device)
    true_mu = true_mu.to(device)
    true_sigma = true_sigma.to(device)

    # =========================
    # Predictor Step
    # =========================
    conv_gaussian_predictor.train()
    mu, sigma = conv_gaussian_predictor(input_texture)
    loss_predictor = criterion_mse(mu, true_mu) + criterion_mse(sigma, true_sigma)
    optimizer_predictor.zero_grad()
    loss_predictor.backward()
    optimizer_predictor.step()

    # =========================
    # Generate Gaussian-like texture
    # =========================
    gaussian_reproducer.train()
    reproduced_texture = gaussian_reproducer(input_texture)

    # =========================
    # Discriminator Step
    # =========================
    discriminator.train()
    real_label = torch.ones((input_texture.size(0), 1), device=device)
    fake_label = torch.zeros((input_texture.size(0), 1), device=device)

    real_output = discriminator(input_texture)
    fake_output = discriminator(reproduced_texture.detach())

    loss_discriminator = criterion_bce(real_output, real_label) + criterion_bce(fake_output, fake_label)
    optimizer_discriminator.zero_grad()
    loss_discriminator.backward()
    optimizer_discriminator.step()

    # =========================
    # Reproducer Step
    # =========================
    gaussian_reproducer.train()
    fake_output_for_reproducer = discriminator(reproduced_texture)
    loss_reproducer = criterion_bce(fake_output_for_reproducer, real_label)
    optimizer_reproducer.zero_grad()
    loss_reproducer.backward()
    optimizer_reproducer.step()

    # =========================
    # Binary Classifier Step
    # =========================
    binary_classifier.train()
    classifier_input = torch.cat((mu, sigma), dim=1)
    is_gaussian = binary_classifier(classifier_input)
    target_is_gaussian = torch.ones((input_texture.size(0), 1), device=device)  # Assume it's Gaussian for training
    loss_classifier = criterion_bce(is_gaussian, target_is_gaussian)
    optimizer_classifier.zero_grad()
    loss_classifier.backward()
    optimizer_classifier.step()

    # =========================
    # Rolling Cache Update
    # =========================
    rolling_cache.add(reproduced_texture.detach().cpu())

    # =========================
    # Return Losses for Logging
    # =========================
    return {
        'loss_predictor': loss_predictor.item(),
        'loss_discriminator': loss_discriminator.item(),
        'loss_reproducer': loss_reproducer.item(),
        'loss_classifier': loss_classifier.item()
    }

# Evaluation Step
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    average_loss = total_loss / len(dataloader.dataset)
    return average_loss

# Training Function
def train_model(num_epochs=20, batch_size=32, save_path='models'):
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare Dataset and DataLoader
    dataset = SyntheticGaussianDataset(num_samples=10000, image_size=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Tracking losses
    history = {
        'predictor': [],
        'discriminator': [],
        'reproducer': [],
        'classifier': []
    }

    for epoch in range(1, num_epochs + 1):
        epoch_losses = {
            'loss_predictor': 0.0,
            'loss_discriminator': 0.0,
            'loss_reproducer': 0.0,
            'loss_classifier': 0.0
        }
        for batch_idx, (inputs, params) in enumerate(dataloader):
            inputs = inputs.to(device)
            true_mu = params[:, 0].unsqueeze(1).to(device)
            true_sigma = params[:, 1].unsqueeze(1).to(device)

            losses = train_step(inputs, true_mu, true_sigma)
            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], "
                      f"Losses: {losses}")

        # Average losses for the epoch
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
            history_key = key.replace('loss_', '')
            history[history_key].append(epoch_losses[key])

        print(f"Epoch [{epoch}/{num_epochs}] Completed. "
              f"Predictor Loss: {epoch_losses['loss_predictor']:.4f}, "
              f"Discriminator Loss: {epoch_losses['loss_discriminator']:.4f}, "
              f"Reproducer Loss: {epoch_losses['loss_reproducer']:.4f}, "
              f"Classifier Loss: {epoch_losses['loss_classifier']:.4f}")

        # Save models at each epoch
        torch.save(conv_gaussian_predictor.state_dict(), os.path.join(save_path, f'conv_gaussian_predictor_epoch{epoch}.pth'))
        torch.save(binary_classifier.state_dict(), os.path.join(save_path, f'binary_classifier_epoch{epoch}.pth'))
        torch.save(gaussian_reproducer.state_dict(), os.path.join(save_path, f'gaussian_reproducer_epoch{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(save_path, f'discriminator_epoch{epoch}.pth'))

    # Plot Loss Curves
    plt.figure(figsize=(12, 8))
    for key, values in history.items():
        plt.plot(range(1, num_epochs + 1), values, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curves.png'))
    plt.show()

    print("Training Completed and Models Saved.")

# Inference Function
def generate_texture(input_texture, save_path='generated_textures'):
    os.makedirs(save_path, exist_ok=True)
    conv_gaussian_predictor.eval()
    gaussian_reproducer.eval()
    binary_classifier.eval()
    discriminator.eval()
    with torch.no_grad():
        mu, sigma = conv_gaussian_predictor(input_texture.to(device))
        reproduced_texture = gaussian_reproducer(input_texture.to(device))
        is_gaussian = binary_classifier(torch.cat((mu, sigma), dim=1))

    # Move tensors to CPU and convert to numpy
    input_texture = input_texture.cpu().squeeze().numpy()
    reproduced_texture = reproduced_texture.cpu().squeeze().numpy()
    mu = mu.cpu().squeeze().item()
    sigma = sigma.cpu().squeeze().item()
    is_gaussian = is_gaussian.cpu().squeeze().item()

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Input Texture')
    plt.imshow(input_texture, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Reproduced Texture')
    plt.imshow(reproduced_texture, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Reproduced Texture with Parameters')
    plt.imshow(reproduced_texture, cmap='gray')
    plt.text(0, 1, f"Mu: {mu:.2f}\nSigma: {sigma:.2f}\nIs Gaussian: {is_gaussian:.2f}",
             color='red', fontsize=12, transform=plt.gca().transAxes)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'generated_texture.png'))
    plt.show()

import torch
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class IsolationMap:
    def __init__(self):
        """
        Central authority for orchestrating network flow and isolation.
        """
        self.graph = nx.DiGraph()
        self.models = {}
        self.optimizers = {}
        self.loss_functions = {}

    def add_model(self, name, model, optimizer=None, loss_function=None):
        """
        Add a model to the IsolationMap.

        Parameters:
        - name (str): Name of the model.
        - model (nn.Module): The PyTorch model.
        - optimizer (torch.optim.Optimizer, optional): Optimizer for the model.
        - loss_function (callable, optional): Loss function associated with the model.
        """
        self.models[name] = model
        if optimizer:
            self.optimizers[name] = optimizer
        if loss_function:
            self.loss_functions[name] = loss_function
        self.graph.add_node(name)
    def update_modes(self, source, target, new_modes):
        if self.graph.has_edge(source, target):
            self.graph.edges[source, target]['modes'] = new_modes
        else:
            raise ValueError(f"No connection exists between {source} and {target}.")
    def forward_safe(self, input_tensor, start_node, target_node):
        try:
            return self.forward(input_tensor, start_node, target_node)
        except Exception as e:
            print(f"Error during forward pass from {start_node} to {target_node}: {e}")
            return None
    def replace_model(self, name, new_model, new_optimizer=None, new_loss_function=None):
        if name in self.models:
            self.models[name] = new_model
            if new_optimizer:
                self.optimizers[name] = new_optimizer
            if new_loss_function:
                self.loss_functions[name] = new_loss_function
        else:
            raise ValueError(f"Model {name} does not exist in IsolationMap.")

    def connect(self, source, target, modes):
        """
        Connect models with directional data flow and isolation modes.

        Parameters:
        - source (str): Name of the source model.
        - target (str): Name of the target model.
        - modes (list[int]): Isolation modes:
            0 -> Reset gradient
            1 -> Clone tensor
            2 -> Detach tensor
            3 -> Destructive unit conversion
        """
        self.graph.add_edge(source, target, modes=modes)

    def isolate_tensor(self, tensor, modes):
        """
        Apply isolation modes to a tensor or a tuple of tensors.

        Parameters:
        - tensor (torch.Tensor or tuple): Input tensor or tuple of tensors.
        - modes (list[int]): Sequence of isolation modes.
        """
        if isinstance(tensor, tuple):
            return tuple((self.isolate_tensor(t, modes) for t in tensor), dim=1)
        
        result = tensor
        for mode in modes:
            if mode == 0:
                result = result.detach().clone().requires_grad_()
            elif mode == 1:
                result = result.clone()
            elif mode == 2:
                result = result.detach()
            elif mode == 3:
                result = torch.tensor(result.cpu().numpy(), device=tensor.device)
            else:
                raise ValueError(f"Unknown isolation mode: {mode}")
        return result

    def forward(self, input_tensor, start_node, target_node):
        """
        Execute the forward pass through the network.

        Parameters:
        - input_tensor (torch.Tensor): Input tensor.
        - start_node (str): Starting model.
        - target_node (str): Target model.

        Returns:
        - torch.Tensor: Output from the target model.
        """
        current_tensor = input_tensor
        path = nx.shortest_path(self.graph, source=start_node, target=target_node)
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            model = self.models[source]
            current_tensor = model(current_tensor)
            modes = self.graph.edges[source, target]['modes']
            current_tensor = self.isolate_tensor(current_tensor, modes)
        return current_tensor

    def train_step(self, input_tensor, labels, start_node, target_node):
        """
        Perform a training step through the map.

        Parameters:
        - input_tensor (torch.Tensor): Input data.
        - labels (torch.Tensor): Ground truth labels.
        - start_node (str): Starting model.
        - target_node (str): Target model.

        Returns:
        - dict: Loss values for all models involved in the path.
        """
        current_tensor = input_tensor
        path = nx.shortest_path(self.graph, source=start_node, target=target_node)
        losses = {}

        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            model = self.models[source]
            optimizer = self.optimizers.get(source)
            loss_fn = self.loss_functions.get(source)

            # Forward pass
            current_tensor = model(current_tensor)
            modes = self.graph.edges[source, target]['modes']
            current_tensor = self.isolate_tensor(current_tensor, modes)

            # Compute loss and backward pass
            if loss_fn and labels is not None:
                loss = loss_fn(current_tensor, labels)
                losses[source] = loss.item()
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return losses
    def visualize_map(self):
        nx.draw(self.graph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.show()

import logging
logging.basicConfig(level=logging.INFO)

def log_updates(message):
    logging.info(message)

def main():
    iso_map = IsolationMap()

    # Add models, optimizers, and loss functions
    iso_map.add_model("Predictor", conv_gaussian_predictor, optimizer_predictor, nn.MSELoss())
    iso_map.add_model("Classifier", binary_classifier, optimizer_classifier, nn.BCELoss())
    iso_map.add_model("Reproducer", gaussian_reproducer, optimizer_reproducer, None)
    iso_map.add_model("Discriminator", discriminator, optimizer_discriminator, nn.BCELoss())

    # Connect models with initial isolation modes
    iso_map.connect("Classifier", "Predictor", [0])
    iso_map.connect("Predictor", "Reproducer", [0])
    iso_map.connect("Classifier", "Discriminator", [0])
    iso_map.connect("Reproducer", "Discriminator", [0])
    iso_map.connect("Discriminator", "Classifier", [0])

    # Main loop
    for iteration in range(100):  # Example: 100 iterations
        try:
            # Dynamic update of isolation modes (example modification)
            if iteration % 10 == 0:
                iso_map.update_modes("Predictor", "Classifier", [0])

            # Input tensor and labels (synthetic example)
            input_tensor = torch.randn(32, 1, 32, 32).to(device)
            labels = torch.randn(32, 1).to(device)

            # Train step with forward pass
            losses = iso_map.train_step(input_tensor, labels, "Predictor", "Discriminator")
            print(f"Iteration {iteration}, Losses: {losses}")

        except Exception as e:
            print(f"Error during iteration {iteration}: {e}")


main()