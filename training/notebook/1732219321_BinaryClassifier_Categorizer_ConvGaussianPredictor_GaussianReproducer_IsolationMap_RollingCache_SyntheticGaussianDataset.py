import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import networkx as nx

# =========================
# Rolling Cache Class
# =========================
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

# =========================
# Networks Definitions
# =========================

# Binary Classifier Network
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(1 * 32 * 32, 128)  # Input is flattened
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output probability

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Convolutional Network to Predict Gaussian Parameters
class ConvGaussianPredictor(nn.Module):
    def __init__(self):
        super(ConvGaussianPredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc_mu = nn.Linear(64, 1)     # Predicts mean
        self.fc_sigma = nn.Linear(64, 1)  # Predicts standard deviation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 16x16
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x))  # Ensure sigma is positive
        return mu, sigma

# Gaussian Reproducer Network
class GaussianReproducer(nn.Module):
    def __init__(self):
        super(GaussianReproducer, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Inputs: mu and sigma
        self.fc2 = nn.Linear(64, 1 * 32 * 32)  # Output is a flattened image

    def forward(self, params):
        x = F.relu(self.fc1(params))
        x = torch.sigmoid(self.fc2(x))  # Ensure output is between 0 and 1
        x = x.view(-1, 1, 32, 32)       # Reshape to image
        return x

# Categorizer Network
class Categorizer(nn.Module):
    def __init__(self):
        super(Categorizer, self).__init__()
        self.fc1 = nn.Linear(5 * 32 * 32, 256)  # Inputs: original, generated, 3 rejected
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Example categorization into 2 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation; assuming CrossEntropyLoss
        return x

# =========================
# IsolationMap Class
# =========================
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
        Apply isolation modes to a tensor.

        Parameters:
        - tensor (torch.Tensor): Input tensor.
        - modes (list[int]): Sequence of isolation modes.
        """
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

# =========================
# Synthetic Gaussian Dataset
# =========================
class SyntheticGaussianDataset(Dataset):
    def __init__(self, num_samples=10000, image_size=32):
        self.num_samples = num_samples
        self.image_size = image_size
        self.data, self.params = self._generate_data()

    def _generate_data(self):
        data = []
        params = []
        for _ in range(self.num_samples):
            is_gaussian = np.random.choice([0, 1])  # 0: Noise, 1: Gaussian
            if is_gaussian:
                mu = np.random.uniform(-1, 1, size=(1,))
                sigma = np.random.uniform(0.5, 1.5, size=(1,))
                grid = np.linspace(-3, 3, self.image_size)
                x, y = np.meshgrid(grid, grid)
                z = np.exp(-((x - mu[0])**2 + (y - mu[0])**2) / (2 * sigma[0]**2))
                z = z / z.max()
                z = z + 0.1 * np.random.randn(*z.shape)  # Add slight noise
                z = np.clip(z, 0, 1)
                data.append(z.astype(np.float32))
                params.append((mu[0], sigma[0]))
            else:
                z = np.random.rand(self.image_size, self.image_size).astype(np.float32)
                data.append(z)
                params.append((0.0, 1.0))  # Dummy parameters
        data = np.expand_dims(np.array(data), axis=1)  # Shape: (N, 1, H, W)
        params = np.array(params, dtype=np.float32)    # Shape: (N, 2)
        return torch.tensor(data), torch.tensor(params)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.params[idx]

# =========================
# Training and Evaluation
# =========================
def train_model(num_epochs=20, batch_size=64, save_path='models', device='cpu'):
    os.makedirs(save_path, exist_ok=True)

    # Initialize IsolationMap
    iso_map = IsolationMap()

    # Instantiate Models
    binary_classifier = BinaryClassifier().to(device)
    conv_gaussian_predictor = ConvGaussianPredictor().to(device)
    gaussian_reproducer = GaussianReproducer().to(device)
    categorizer = Categorizer().to(device)

    # Define Optimizers
    optimizer_classifier = optim.Adam(binary_classifier.parameters(), lr=0.001)
    optimizer_predictor = optim.Adam(conv_gaussian_predictor.parameters(), lr=0.001)
    optimizer_reproducer = optim.Adam(gaussian_reproducer.parameters(), lr=0.001)
    optimizer_categorizer = optim.Adam(categorizer.parameters(), lr=0.001)

    # Define Loss Functions
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    # Add models to IsolationMap
    iso_map.add_model("BinaryClassifier", binary_classifier, optimizer_classifier, criterion_bce)
    iso_map.add_model("ConvGaussianPredictor", conv_gaussian_predictor, optimizer_predictor, criterion_mse)
    iso_map.add_model("GaussianReproducer", gaussian_reproducer, optimizer_reproducer, None)
    # Note: Categorizer will be handled separately as it has multiple inputs

    # Connect models with isolation modes
    iso_map.connect("BinaryClassifier", "ConvGaussianPredictor", [2])  # Detach tensor
    iso_map.connect("ConvGaussianPredictor", "GaussianReproducer", [1])  # Clone tensor

    # Initialize Rolling Cache
    rolling_cache = RollingCache(max_size=3)  # Example size

    # Prepare Dataset and DataLoader
    dataset = SyntheticGaussianDataset(num_samples=10000, image_size=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Tracking losses
    history = {
        'binary_classifier': [],
        'conv_predictor': [],
        'categorizer': []
    }

    for epoch in range(1, num_epochs + 1):
        binary_classifier.train()
        conv_gaussian_predictor.train()
        gaussian_reproducer.train()
        categorizer.train()

        epoch_losses = {
            'binary_classifier': 0.0,
            'conv_predictor': 0.0,
            'categorizer': 0.0
        }

        for batch_idx, (inputs, params) in enumerate(dataloader):
            inputs = inputs.to(device)      # Shape: (B, 1, 32, 32)
            true_mu = params[:, 0].unsqueeze(1).to(device)     # Shape: (B, 1)
            true_sigma = params[:, 1].unsqueeze(1).to(device)  # Shape: (B, 1)

            # =========================
            # Binary Classifier Step
            # =========================
            optimizer_classifier.zero_grad()
            binary_output = binary_classifier(inputs)  # Shape: (B, 1)
            # Labels: 1 for Gaussian, 0 for Noise
            is_gaussian = (params[:,0] != 0).float().unsqueeze(1).to(device)  # Assuming noise has mu=0
            loss_classifier = criterion_bce(binary_output, is_gaussian)
            loss_classifier.backward()
            optimizer_classifier.step()

            # Accumulate loss
            epoch_losses['binary_classifier'] += loss_classifier.item()

            # Determine which samples are Gaussian
            gaussian_indices = is_gaussian.squeeze() > 0.5
            noise_indices = is_gaussian.squeeze() <= 0.5

            # =========================
            # ConvGaussianPredictor Step (Only for Gaussian samples)
            # =========================
            if gaussian_indices.sum() > 0:
                gaussian_inputs = inputs[gaussian_indices]
                gaussian_true_mu = true_mu[gaussian_indices]
                gaussian_true_sigma = true_sigma[gaussian_indices]

                optimizer_predictor.zero_grad()
                pred_mu, pred_sigma = conv_gaussian_predictor(gaussian_inputs)
                loss_predictor = criterion_mse(pred_mu, gaussian_true_mu) + criterion_mse(pred_sigma, gaussian_true_sigma)
                loss_predictor.backward()
                optimizer_predictor.step()

                # Accumulate loss
                epoch_losses['conv_predictor'] += loss_predictor.item()

                # =========================
                # Gaussian Reproducer Step
                # =========================
                optimizer_reproducer.zero_grad()
                # Prepare parameters for reproducer
                params_reproducer = torch.cat((pred_mu, pred_sigma), dim=1)
                reproduced_textures = gaussian_reproducer(params_reproducer)

                # Add to Rolling Cache
                for tex in reproduced_textures.detach().cpu():
                    rolling_cache.add(tex)

            # =========================
            # Categorizer Step
            # =========================
            # Prepare inputs for Categorizer
            # Original textures
            original_textures = inputs
            # Generated textures
            if gaussian_indices.sum() > 0:
                generated_textures = reproduced_textures.detach().cpu()
                # Pad generated_textures if batch has noise
                if noise_indices.sum() > 0:
                    generated_textures = torch.cat([generated_textures, torch.zeros(noise_indices.sum(), 1, 32, 32)], dim=0)
            else:
                generated_textures = torch.zeros(inputs.size(0), 1, 32, 32).cpu()

            # Rejected textures from Rolling Cache
            rejected_textures = torch.stack(rolling_cache.get_all()) if len(rolling_cache.get_all()) > 0 else torch.zeros(0, 1, 32, 32)
            if rejected_textures.size(0) < 3:
                padding = torch.zeros(3 - rejected_textures.size(0), 1, 32, 32)
                rejected_textures = torch.cat([rejected_textures, padding], dim=0)
            else:
                rejected_textures = rejected_textures[:3]

            rejected_textures = rejected_textures.unsqueeze(0).repeat(inputs.size(0), 1, 1, 1, 1)  # Shape: (B, 3, 1, 32, 32)
            rejected_textures = rejected_textures.view(inputs.size(0), -1, 32, 32).to(device)  # Shape: (B, 3, 32, 32)

            # New noise
            new_noise = torch.rand_like(inputs).to(device)

            # New Gaussian textures (using random mu and sigma)
            rand_mu = torch.randn(inputs.size(0), 1).to(device)
            rand_sigma = torch.abs(torch.randn(inputs.size(0), 1)).to(device)
            grid = torch.linspace(-3, 3, 32).to(device)
            x, y = torch.meshgrid(grid, grid)
            x = x.unsqueeze(0).expand(inputs.size(0), -1, -1)
            y = y.unsqueeze(0).expand(inputs.size(0), -1, -1)
            new_gaussian = torch.exp(-((x - rand_mu.unsqueeze(2))**2 + (y - rand_mu.unsqueeze(2))**2) / (2 * rand_sigma.unsqueeze(2)**2))
            new_gaussian = new_gaussian / new_gaussian.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            new_gaussian = new_gaussian.unsqueeze(1) + 0.1 * torch.randn_like(x).unsqueeze(1)
            new_gaussian = torch.clamp(new_gaussian, 0, 1)

            # Concatenate all inputs for Categorizer
            # Shape for Categorizer: (B, 5, 32, 32) -> original, generated, 3 rejected
            if rejected_textures.size(1) < 3:
                padding = torch.zeros(inputs.size(0), 3 - rejected_textures.size(1), 32, 32).to(device)
                rejected_textures = torch.cat([rejected_textures, padding], dim=1)
            categorizer_input = torch.cat([
                original_textures,
                generated_textures,
                rejected_textures
            ], dim=1)  # Shape: (B, 5, 32, 32)

            # Forward pass through Categorizer
            optimizer_categorizer.zero_grad()
            categorizer_output = categorizer(categorizer_input)  # Shape: (B, 2)
            # Example labels: 0 for noise, 1 for Gaussian
            categorizer_labels = is_gaussian.long()
            loss_categorizer = criterion_ce(categorizer_output, categorizer_labels)
            loss_categorizer.backward()
            optimizer_categorizer.step()

            # Accumulate loss
            epoch_losses['categorizer'] += loss_categorizer.item()

        # Average losses for the epoch
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
            history[key].append(epoch_losses[key])

        print(f"Epoch [{epoch}/{num_epochs}] Completed. "
              f"Binary Classifier Loss: {epoch_losses['binary_classifier']:.4f}, "
              f"Conv Predictor Loss: {epoch_losses['conv_predictor']:.4f}, "
              f"Categorizer Loss: {epoch_losses['categorizer']:.4f}")

        # Save models at each epoch
        torch.save(binary_classifier.state_dict(), os.path.join(save_path, f'binary_classifier_epoch{epoch}.pth'))
        torch.save(conv_gaussian_predictor.state_dict(), os.path.join(save_path, f'conv_gaussian_predictor_epoch{epoch}.pth'))
        torch.save(gaussian_reproducer.state_dict(), os.path.join(save_path, f'gaussian_reproducer_epoch{epoch}.pth'))
        torch.save(categorizer.state_dict(), os.path.join(save_path, f'categorizer_epoch{epoch}.pth'))

    # Plot Loss Curves
    def plot_losses(history, save_path='models'):
        plt.figure(figsize=(12, 8))
        for key, values in history.items():
            plt.plot(range(1, len(values)+1), values, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'loss_curves.png'))
        plt.show()

    # =========================
    # Inference Function
    # =========================
    def generate_texture(binary_classifier, conv_predictor, reproducer, categorizer, input_texture, rolling_cache, device='cpu', save_path='generated_textures'):
        os.makedirs(save_path, exist_ok=True)
        binary_classifier.eval()
        conv_predictor.eval()
        reproducer.eval()
        categorizer.eval()
        with torch.no_grad():
            input_texture = input_texture.to(device)
            binary_output = binary_classifier(input_texture)
            is_gaussian = binary_output.squeeze() > 0.5

            if is_gaussian:
                mu, sigma = conv_predictor(input_texture)
                params_reproducer = torch.cat((mu, sigma), dim=1)
                reproduced_texture = reproducer(params_reproducer)
                generated = reproduced_texture.cpu()
            else:
                generated = torch.zeros_like(input_texture).cpu()

            # Get rejected textures
            rejected = torch.stack(rolling_cache.get_all()) if len(rolling_cache.get_all()) > 0 else torch.zeros(0, 1, 32, 32)
            if rejected.size(0) < 3:
                padding = torch.zeros(3 - rejected.size(0), 1, 32, 32)
                rejected = torch.cat([rejected, padding], dim=0)
            else:
                rejected = rejected[:3]

            # Prepare Categorizer input
            rejected = rejected.unsqueeze(0).repeat(input_texture.size(0), 1, 1, 1, 1)  # Shape: (1, 3, 1, 32, 32)
            rejected = rejected.view(input_texture.size(0), -1, 32, 32)  # Shape: (1, 3, 32, 32)
            categorizer_input = torch.cat([input_texture.cpu(), generated, rejected], dim=1)  # Shape: (1, 5, 32, 32)

            # Forward pass through Categorizer
            categorizer_output = categorizer(categorizer_input)
            _, predicted = torch.max(categorizer_output, 1)

        # Move tensors to CPU and convert to numpy
        input_texture_np = input_texture.cpu().squeeze().numpy()
        generated_np = generated.squeeze().numpy()

        # Plotting
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Input Texture')
        plt.imshow(input_texture_np, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Generated Texture')
        plt.imshow(generated_np, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Categorization')
        plt.imshow(generated_np, cmap='gray')
        plt.text(0, 1, f"Is Gaussian: {is_gaussian.item():.2f}\nCategory: {predicted.item()}",
                 color='red', fontsize=12, transform=plt.gca().transAxes)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'generated_texture.png'))
        plt.show()

    # =========================
    # Main Execution
    # =========================
    if __name__ == "__main__":
        # Device Configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Training Parameters
        NUM_EPOCHS = 20
        BATCH_SIZE = 64
        SAVE_PATH = 'models'

        # Start Training
        train_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, save_path=SAVE_PATH, device=device)

        # Plot Loss Curves
        # Note: To plot losses, you might want to save 'history' outside the train_model function or adjust accordingly.

        # Example Inference
        # Load a sample from the dataset
        dataset = SyntheticGaussianDataset(num_samples=1, image_size=32)
        sample_input, sample_params = dataset[0]
        sample_input = sample_input.unsqueeze(0)  # Add batch dimension

        # Load trained models
        binary_classifier = BinaryClassifier().to(device)
        conv_gaussian_predictor = ConvGaussianPredictor().to(device)
        gaussian_reproducer = GaussianReproducer().to(device)
        categorizer = Categorizer().to(device)

        epoch_to_load = NUM_EPOCHS  # Load the last epoch
        binary_classifier.load_state_dict(torch.load(os.path.join(SAVE_PATH, f'binary_classifier_epoch{epoch_to_load}.pth')))
        conv_gaussian_predictor.load_state_dict(torch.load(os.path.join(SAVE_PATH, f'conv_gaussian_predictor_epoch{epoch_to_load}.pth')))
        gaussian_reproducer.load_state_dict(torch.load(os.path.join(SAVE_PATH, f'gaussian_reproducer_epoch{epoch_to_load}.pth')))
        categorizer.load_state_dict(torch.load(os.path.join(SAVE_PATH, f'categorizer_epoch{epoch_to_load}.pth')))

        # Initialize Rolling Cache
        rolling_cache = RollingCache(max_size=3)

        # Generate Texture
        generate_texture(binary_classifier, conv_gaussian_predictor, gaussian_reproducer, categorizer, sample_input, rolling_cache, device=device, save_path='generated_textures')

        print("Inference Completed and Texture Generated.")
