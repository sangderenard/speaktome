#!/usr/bin/env python3
"""FontMapper FM38 utility functions."""
from __future__ import annotations

# --- END HEADER ---

# --- Core Imports (always required) ---
try:
    import sys
    import os
    import math
    import time
    import random
    import struct
    import json
    import threading
    import datetime
    import queue
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import transforms
    from torchvision.transforms import functional as TF
    from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops, ImageSequence
    from fontTools.ttLib import TTFont
    import yaml
    from collections import defaultdict
except Exception:
    import sys
    print(f"Optional dependencies missing in {__file__} -> {sys.exc_info()[1]}")
# --- END HEADER ---

from .optional_dependencies import (
    pika,
    PIKA_AVAILABLE,
    qtwidgets,
    qtgui,
    qtcore,
    PYQT_AVAILABLE,
    colorsys,
    ssim,
    COLOR_MIXING_AVAILABLE,
    pynvml,
    NVML_AVAILABLE,
)

channel = None

# --- END IMPORT WALL ---

from .modules import (
    ModelCompatConfig,
    ModelConfig,
    CharSorter,
    AddRandomNoise,
    DistortionChain,
    RandomGaussianBlur,
    ToTensorAndToDevice,
    CustomDataset,
    CustomInputDataset,
)

device_id = 0
# This function gets the free memory for a specific GPU
def get_gpu_memory_utilization(device_id):
    if NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_percentage = (info.free / info.total) * 100
        return 100 - free_percentage
    return 0

def getconfig():
    server = False
    user_pil = None
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: script.py <config_file> [image_file]")
        sys.exit(1)
    
    config_file_path = sys.argv[1]
    config_extension = os.path.splitext(config_file_path)[1].lower()
    
    if config_extension != '.yaml':
        print("The first argument must be a YAML configuration file.")
        sys.exit(1)
    
    print("Configuration file detected and processed.")
    # Here you would add code to process the configuration file
    
    if len(sys.argv) == 3:
        if sys.argv[2] == 'server':
            server = True
        else:
            image_file_path = sys.argv[2]
            image_extension = os.path.splitext(image_file_path)[1].lower()
            
            if image_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                try:
                    # Attempt to open the image to see if it's loadable by PIL
                    with Image.open(image_file_path) as img:
                        user_pil = img.copy()
                except IOError:
                    print("Failed to open the image file.")
                    sys.exit(1)
            else:
                print("Unsupported image file type.")
                sys.exit(1)
        
    return config_file_path, user_pil, server

config_file, user_pil, server = getconfig()

override_outer_learning = override_alternating_outer_learning = override_homework_epochs = override_category_epochs = override_gradient_epochs = override_image_epochs = override_demo_epochs = override_outer_lr = override_category_lr = override_gradient_lr = override_image_lr = override_demo_lr = override_meta_model_learning = override_complexity = skip_refreshments = None
device = font_size = charset = font_files = batch_size = model_name = model_base = conv1_out = conv2_out = linear_out = dropout = learning_rate = epochs = None
epochs_per_preview = chars_per_preview = interactive = model_path = demo_image = image_width = image_height = demo_images = live = complexity = training_loss_function = None
demo_loss_function = gradient_loss_function = training_gradients = training_categories = training_images = image_batch_limit = refresh_rate = quick_test = None
pika_messaging = pika_username = pika_password = pika_password_file = hitl = human_mode = None

metamodel_path = general_model_path = training_images_path = implemented_shapes = None
quick_test = training_images = training_gradients = training_categories = image_batch_limit = live = TRAINING_IMAGES_DIRECTORY = None
max_size = interactivemode = training_loss_function = demo_loss_function = gradient_loss_function = font_size = font_files = None
batch_size = model_base = model_name = conv1_out = conv2_out = linear_out = dropout = learning_rate = epochs = epochs_per_preview = None
chars_per_preview = model_path_override = demo_image = demo_images = refresh_rate = hitl = demo_image = COMPLEXITY_LEVEL = None

config_to_program_var_names = {
  "model_path": 'model_path_override',
  "interactive": "interactivemode",
}
def load_config(config_file):
    globals()['demo_image'] = None
    with open(config_file, 'r') as f:
        options_config = yaml.safe_load(f)

    tensor_list = options_config['tensor_list']
    for name, value in tensor_list.items():
        globals()[name] = torch.tensor(float(value), requires_grad=True)
    for name, value in options_config.items():
        if name == 'tensor_list':
            continue
        elif name in config_to_program_var_names:
            name = config_to_program_var_names[name]

        if name == "demo_image" and value != "":
            globals()[name] = Image.open(value)
        elif name == "device":
            if server:
                globals()[name] = torch.device('cpu')
            else:
                globals()[name] = torch.device(value if torch.cuda.is_available() else 'cpu')
        elif name == "max_size":
            globals()[name] = ( value[0], value[1] )
        else:
            globals()[name] = value

load_config(config_file)
print(globals())
def read_password_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: Password file {filename} not found.")
        return None

def pika_keepalive(channel):
    """Function to send a heartbeat to RabbitMQ to keep the connection alive."""
    if not PIKA_AVAILABLE:
        return None
    try:
        while True:
            channel.basic_publish(
                exchange='',
                routing_key='ascii_statistics_queue',
                body=json.dumps({'Keepalive message': 'Keepalive message'})
            )
            print("Keepalive message sent")
            time.sleep(10)
    except pika.exceptions.AMQPConnectionError as e:
        print(f"Failed to send to RabbitMQ: {e}")
        return None

def initialize_pika_connection(username, password):
    if not PIKA_AVAILABLE:
        return None
    try:
        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost', 5672, '/', credentials)
        )
        channel = connection.channel()
        channel.queue_declare(queue='ascii_statistics_queue')
        return channel
    except pika.exceptions.AMQPConnectionError as e:
        print(f"Failed to connect to RabbitMQ: {e}")
        return None
pika_messaging = pika_messaging
if pika_messaging and PIKA_AVAILABLE:
    username = pika_username
    password = pika_password if pika_password else read_password_from_file(pika_password_file)
    
    if username and password:
        channel = initialize_pika_connection(username, password)
        if channel:
            globals()['channel'] = channel
            keepalive_thread = threading.Thread(target=pika_keepalive, args=(channel,))
            keepalive_thread.daemon = True
            keepalive_thread.start()
            print("Pika messaging is initialized and running.")
        else:
            print("Failed to initialize Pika messaging.")
            pika_messaging = False
    else:
        pika_messaging = False
        print("Username or password not provided or could not be read from the file.")
else:
    pika_messaging = False


def pika_message(start_time, outer_epoch, teaching_epoch, epoch, subject, inner_epoch, outer_learning, learning_rate, total_loss, maximum_memory_utilization, pika_messaging):
    if not (PIKA_AVAILABLE and pika_messaging):
        return start_time
    pika_start = time.time()
    try:
        message_data = {
            'outer_epoch': outer_epoch + 1,
            'teaching_epoch': teaching_epoch + 1,
            'homework_epoch': epoch + 1,
            'subject': subject,
            'subject_epoch': inner_epoch + 1, 
            'outer_learning': outer_learning.item(),
            'learning_rate': learning_rate,
            'batch_total_loss': total_loss.item(),
            'maximum_memory_utilization': maximum_memory_utilization
        }

        # Dynamically print all message details
        print("Sending message details:")

        for key, value in message_data.items():
            print(f"{key}: {value}")
        if pika_messaging:
            message = json.dumps(message_data)
            
            channel.basic_publish(exchange='', routing_key='ascii_statistics_queue', body=message)

    except pika.exceptions.AMQPConnectionError as e:
        print(f"Failed to send to RabbitMQ: {e}")
        return None
    pika_end = time.time()          
    pika_time = pika_end - pika_start
    start_time = start_time + pika_time
    
    return start_time



FMversion = "FM37"


class ModelCompatConfig:
    def __init__(self, charset, font_files, font_size, conv1_out, conv2_out, linear_out, width, height):
        self.charset = charset
        self.width = width
        self.height = height
        self.font_files = font_files
        self.font_size = font_size
        self.conv1_out = conv1_out
        self.conv2_out = conv2_out
        self.linear_out = linear_out
        
class ModelConfig:
    def __init__(self, compatibility_model, dropout, learning_rate, epochs, batch_size, demo_image=None, epochs_per_preview=10000000, model_path="model"):
        self.charset = compatibility_model.charset
        self.refresh_rate = refresh_rate
        self.demo_image = demo_image
        self.demo_images = demo_images
        self.training_categories = training_categories
        self.training_gradients = training_gradients
        self.training_images = training_images
        self.human_in_the_loop = hitl
        self.image_batch_limit = image_batch_limit
        self.epochs_per_preview = epochs_per_preview
        self.font_files = compatibility_model.font_files
        self.font_size = compatibility_model.font_size
        self.conv1_out = compatibility_model.conv1_out
        self.conv2_out = compatibility_model.conv2_out
        self.linear_out = compatibility_model.linear_out
        self.width = compatibility_model.width
        self.height = compatibility_model.height
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.dataloader = None
        self.gradient_loss_function = gradient_loss_function
        self.training_loss_function = training_loss_function
        self.demo_loss_function = demo_loss_function
        self.version = f"{font_size}-{conv1_out}-{conv2_out}-{linear_out}-{dropout}-{learning_rate}-prev-{model_base}"
        self.model_path = f"{FMversion}--{self.version}--{model_name}"
        if model_path_override != "":
            self.model_path = model_path_override


import inspect
some_model_directory = "./"

def display_image(config, image_list, out_image=False):
    with torch.no_grad():
        for queue_image in image_list:
            queue_image_ascii = pilToASCII(queue_image, config, max_size= queue_image.size)
            print(queue_image_ascii)
            if out_image:
                queue_image_text_rendering = render_text_lines(config.charBitmasks, queue_image_ascii, config.charset, config.width, config.height)
                queue_image_image = tensor_to_image(queue_image_text_rendering)
                return queue_image_image

def save_model(config, path=None, filename=None, meta_model=None):
    """Saves the model's state dictionary."""
    if config == None:
        torch.save(meta_model.state_dict(), metamodel_path)
        print(f"MetaModel saved.")
    else:
        if path is None and filename is None:
            path = os.path.join(some_model_directory, f"{config.model_path}.pth")
        if filename is None:
            torch.save(config.model.state_dict(), path)
            print(f"Model saved to: {path}")
        else:
            torch.save(config.model.state_dict(), filename)
            print(f"Model saved to: {filename}")
import random
import struct
import math

def generate_task_pattern(meta_model_output, num_training_tasks, interest_offset = 12, max_length=10):
    # Avoid the highest order bits likely to have little variety with the interest_offset
    # Scale the maximum length possible to be generated with max_length

    # Rescale the normalized input to give variety to the high order bits.
    meta_model_output = meta_model_output * (2 ** 20)
    
    # Convert the float to a 64-bit binary string
    packed_float = struct.pack('!d', meta_model_output)
    unpacked_float = struct.unpack('!Q', packed_float)[0]
    float_binary_full = format(unpacked_float, '064b')

    # Trim leading zeros and manage all-zero case
    trimmed_binary = float_binary_full.lstrip('0') or '1010' * (64 / 4)

    # Extract the first 4 bits as an integer and use them as a multiplier to determine total length
    length_bits = trimmed_binary[interest_offset:interest_offset+4] if len(trimmed_binary) >= 4 + interest_offset else '1010'
    length_scale = (int(length_bits, 2) + num_training_tasks)

    # Calculate number of bits needed per index and the total pattern length by taking the remainder of the scaled length divided by the max_length
    num_bits_needed = max(1, math.ceil(math.log2(num_training_tasks)))
    pattern_length = length_scale % (max_length - num_training_tasks) + num_training_tasks

    # Create a binary pattern long enough to form the pattern
    remainder_binary = trimmed_binary[interest_offset:]
    needed_bits_total = num_bits_needed * pattern_length
    full_length_binary = (remainder_binary * ((needed_bits_total // len(remainder_binary)) + 1))[:needed_bits_total]

    # Convert the full length binary string into indices
    task_pattern = [map_binary_to_int(int(full_length_binary[i:i+num_bits_needed], 2), num_bits_needed, num_training_tasks) for i in range(0, len(full_length_binary), num_bits_needed)]

    # Ensure all indices are represented at least once
    complete_set = set(range(num_training_tasks))
    current_set = set(task_pattern)
    missing_indices = complete_set - current_set

    while missing_indices:
        most_frequent_index = max(current_set, key=task_pattern.count)
        replace_index = random.choice([i for i, x in enumerate(task_pattern) if x == most_frequent_index])
        missing_index = missing_indices.pop()
        task_pattern[replace_index] = missing_index
        current_set.add(missing_index)
    # Convert list to a PyTorch tensor and enable gradient tracking.
    task_pattern_tensor = torch.tensor(task_pattern, dtype=torch.float32, requires_grad=True)

    return task_pattern_tensor

def map_binary_to_int(value, bit_pattern_size, num_outputs):
    """
    This function efficiently maps integers represented by a specified number of bits 
    (bit_pattern_size) to a desired range of outputs (num_outputs) while aiming for 
    uniform probability distribution across the entire output range. 

    The approach avoids using modulo operations, which favor the initial outputs, 
    or purely random selection, which leads to uneven distribution. Instead, it leverages 
    probabilistic mapping to achieve a continuous and sequential mapping of input 
    values to output values with equal probability.

    Args:
        value: The integer value represented by the bit pattern.
        bit_pattern_size: The number of bits in the bit pattern.
        num_outputs: The desired number of possible outputs.

    Returns:
        The mapped integer within the specified output range.
    """

    max_representable = 2**bit_pattern_size  # Maximum representable input value
    pigeonhole_size = max_representable / num_outputs  # Size of each output range

    value = value + 1  # Adjust for 1-based indexing

    # Create output ranges (pigeonholes)
    pigeonholes = []
    for i in range(num_outputs):
        pigeonholes.append((i * pigeonhole_size, i * pigeonhole_size + pigeonhole_size))

    # Randomly select a point within the input range
    random_point = value - 0.5 + random.random() - 0.5

    # Determine the pigeonhole for the random point
    selected_pigeonhole = None
    for idx, (start, end) in enumerate(pigeonholes):
        idx = idx + 1  # Adjust for 1-based indexing
        if start < random_point < end:
            selected_pigeonhole = idx
            break
        # Handle edge cases where the point falls on a boundary
        elif random_point in (start, end): 
            selected_pigeonhole = random.choice([idx - 1 if idx > 1 else idx, idx + 1 if idx < len(pigeonholes) else idx])
            break

    return selected_pigeonhole - 1 
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward pass is just rounding
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass lets gradients pass through unchanged
        return grad_output
def use_tensor_as_index(tensor, array, value=None):
    # Apply the RoundSTE function
    rounded_indices = RoundSTE.apply(tensor)
    rounded_indices = rounded_indices.long()  # Ensure indices are in long format for indexing
    # Use the rounded indices to index an array
    if value:
        array[rounded_indices] = value
    return array[rounded_indices]
class MetaModel(nn.Module):
    def __init__(self, input_features=15, hidden_dim=128, dropout_layer_one=256, dropout_layer_two=256, num_layers=12, output_features=15):
        super(MetaModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.hidden_layer1 = nn.Linear(hidden_dim, dropout_layer_one)
        self.dropout1 = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(dropout_layer_one)  # Using LayerNorm which is more appropriate here
        
        self.hidden_layer2 = nn.Linear(dropout_layer_one, dropout_layer_two )
        self.dropout2 = nn.Dropout(0.2)
        self.layer_norm2 = nn.LayerNorm(dropout_layer_two)  # Using LayerNorm
        
        self.output_layer = nn.Linear(dropout_layer_two, output_features)
        self.alternator = 0
        self.input_sequence = []

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.view(1, 1, -1)
        self.input_sequence.append(x)
        
        inputs_tensor = torch.cat(self.input_sequence, dim=1) if len(self.input_sequence) > 1 else torch.cat([x, x], dim=1)
        
        lstm_out, _ = self.lstm(inputs_tensor)
        
        # Use only the output from the last time step of LSTM
        x = lstm_out[:, -1, :]  # shape [1, hidden_dim]
        
        x = self.hidden_layer1(x)
        x = self.dropout1(x)
        x = self.layer_norm1(x)  # Apply layer norm correctly
        
        x = self.hidden_layer2(x)
        x = self.dropout2(x)
        x = self.layer_norm2(x)
        
        x = torch.sigmoid(self.output_layer(x))
        return x


    def reset_sequence(self):
        """Resets the input sequence list to start fresh for new sequence processing."""
        self.input_sequence = []
    def interpret_output(self, output, training_tasks, demo_image_count = 1, lr_max=0.001):
        num_training_tasks = len(training_tasks)
        task_to_index = {task: index for index, task in enumerate(training_tasks)}

        def soft_counts(tensor, num_classes):
            # Create a one-hot-like tensor where each value contributes to all indices with a weight based on distance
            contributions = torch.nn.functional.one_hot(tensor.long(), num_classes=num_classes).float()
            # Summing up contributions to get a soft count
            counts = contributions.sum(0)
            return counts



        self.alternator += 1
        if output.dim() > 1:
            output = output.squeeze(0)
        training_pattern = generate_task_pattern(output[12], num_training_tasks, interest_offset = 12, max_length=10)

        # Differentiable counting
        counts = soft_counts(training_pattern, num_training_tasks)

        # Mapping counts to divisors
        pattern_based_divisors = {i: counts[i] for i in range(num_training_tasks)}
        category_divisor = max(1, pattern_based_divisors.get(task_to_index.get('identification', -1), 1))
        gradient_divisor = max(1, pattern_based_divisors.get(task_to_index.get('gradients', -1), 1))
        shapes_divisor = max(1, pattern_based_divisors.get(task_to_index.get('shapes', -1), 1))
        demo_divisor = max(1, pattern_based_divisors.get(task_to_index.get('user images', -1), 1))
        human_divisor = max(1, pattern_based_divisors.get(task_to_index.get('human in the loop', -1), 1))
        
        

        complexity_level = output[11] * 5 if not override_complexity else override_complexity
        outer_learning = (output[0] > 0.5 if not override_outer_learning else override_outer_learning > 0.5) if not override_alternating_outer_learning else torch.tensor(float((self.alternator + (output[0] * 2)) % 2), requires_grad=True) > 0.5
        num_epochs = output[1] * 25 if not override_homework_epochs else override_homework_epochs
        num_category_epochs = output[2] * 80/(complexity_level+category_divisor) if not override_category_epochs else override_category_epochs
        num_gradient_epochs = output[3] * 40/(complexity_level+gradient_divisor) if not override_gradient_epochs else override_gradient_epochs
        num_image_epochs = output[4] * 40/(complexity_level+shapes_divisor) if not override_image_epochs else override_image_epochs
        num_demo_epochs = output[5] * 40/(demo_image_count+demo_divisor) if not override_demo_epochs else override_demo_epochs
        num_human_epochs = output[13] * 10/human_divisor

        override_minimum_outer_learning_epochs = override_minimum_inner_epochs = 0#1

        if override_minimum_inner_epochs:
            num_category_epochs = num_category_epochs - 1
            num_gradient_epochs = num_gradient_epochs - 1
            num_image_epochs = num_image_epochs - 1
            num_demo_epochs = num_demo_epochs -1

        if override_minimum_outer_learning_epochs:
            num_epochs = num_epochs - 1

        if quick_test:
            num_category_epochs = num_category_epochs / quick_test
            num_gradient_epochs = num_gradient_epochs / quick_test
            num_image_epochs = num_image_epochs / quick_test
            num_demo_epochs = num_demo_epochs / quick_test
            num_epochs = num_epochs / 2

        # Adjust learning rates using exponential scale of 10^(-value*6)
        outer_learning_rate = 10 ** (-output[6] * 6) * lr_max if not override_outer_lr else override_outer_lr
        category_learning_rate = 10 ** (-output[7] * 6) * lr_max if not override_category_lr else override_category_lr
        gradient_learning_rate = 10 ** (-output[8] * 6) * lr_max if not override_gradient_lr else override_gradient_lr
        image_learning_rate = 10 ** (-output[9] * 6) * lr_max if not override_image_lr else override_image_lr
        demo_learning_rate = 10 ** (-output[10] * 6) * lr_max if not override_demo_lr else override_demo_lr
        human_learning_rate = 10 ** (-output[14] * 6) * lr_max

        return (training_pattern, complexity_level, outer_learning, num_epochs, num_category_epochs, num_gradient_epochs,
                num_image_epochs, num_demo_epochs, num_human_epochs, outer_learning_rate.item(),
                category_learning_rate.item(), gradient_learning_rate.item(),
                image_learning_rate.item(), demo_learning_rate.item(), human_learning_rate.item())

    
def weighted_average(loss_dataloaders_list):
    if not loss_dataloaders_list:
        return None  # No data to process

    total_weighted_loss = torch.tensor(0.0, requires_grad=True, device=device)
    total_dataloaders = 0

    # Compute the sum of weighted losses and the sum of dataloaders
    for loss_dataloaders in loss_dataloaders_list:
        for loss_value, dataloaders in loss_dataloaders.items():
            total_weighted_loss = total_weighted_loss + loss_value / dataloaders
            total_dataloaders += dataloaders

    if total_dataloaders == 0:
        return torch.Tensor(10.0, requires_grad = True, device = device)

    # Return a tensor with the correct device and gradient context
    weighted_loss_average = total_weighted_loss / len(loss_dataloaders_list)
    return weighted_loss_average

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run_training_loop(maximum_memory_utilization, task_type, optimizer, learning_rate, num_epochs, dataloaders, loss_function, loss_alpha, original_images, outer_learning, config, live, training_size, start_time, outer_epoch, teaching_epoch, epoch, pika_messaging, memory_threshold):
    total_task_loss = torch.tensor(0.0, device=device, requires_grad=True)
    adjust_learning_rate(optimizer, learning_rate)
    total_and_dataloaders_per_inner_epoch = []
    for inner_epoch in range(torch.round(num_epochs).int() + 1):
        completed_dataloaders = torch.tensor(0.0, requires_grad=True)
        if get_gpu_memory_utilization(0) > memory_threshold and inner_epoch > 0:
            break
        total_task_loss = torch.tensor(0.0, device=device, requires_grad=True)


        for index, dataloader in enumerate(dataloaders):
            batch_task_loss = []
            completed_dataloaders = completed_dataloaders + 1
            if task_type == 'human in the loop':
                selected_labels_logits, actual_labels = human_training(config, dataloader)
                batch_task_loss.append(loss_alpha * loss_function(selected_labels_logits, actual_labels))
            elif task_type == 'identification':
                for i, (images, labels) in enumerate(dataloader):
                    outputs = config.model(images)
                    character_identification_visualization(live, outputs, images, labels, config)
                    task_loss = loss_alpha * loss_function(outputs, labels)
                    batch_task_loss.append(task_loss)
            else:
                ascii_art_images = dataloaderImageToImage(dataloader, config, live, training_size)
                for ascii_tensor, original_tensor in zip(ascii_art_images, original_images):
                    resized_evaluated_image = ascii_tensor.squeeze(0)/255.0
                    resized_demo_image = original_tensor.squeeze(0)
                    task_loss = loss_alpha * loss_function(resized_demo_image, resized_evaluated_image)
                    batch_task_loss.append(task_loss)

            averaged_task_loss = sum(batch_task_loss) / len(batch_task_loss)
            total_task_loss = total_task_loss + averaged_task_loss

            current_memory_utilization = get_gpu_memory_utilization(device_id)
            maximum_memory_utilization = max(current_memory_utilization, maximum_memory_utilization)
            if current_memory_utilization > memory_threshold:
                break
        total_and_dataloaders_per_inner_epoch.append({total_task_loss.item(): completed_dataloaders.item()})
        if not outer_learning:
            if torch.round(completed_dataloaders).int() >= 1:
                total_task_loss = total_task_loss / completed_dataloaders
                start_time = pika_message(start_time, outer_epoch, teaching_epoch, epoch, task_type, inner_epoch, outer_learning, learning_rate, total_task_loss, maximum_memory_utilization, pika_messaging)
                optimizer.zero_grad()
                total_task_loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

    total_task_loss = weighted_average(total_and_dataloaders_per_inner_epoch)
    if outer_learning:
        start_time = pika_message(start_time, outer_epoch, teaching_epoch, epoch, task_type, inner_epoch, outer_learning, learning_rate, total_task_loss, maximum_memory_utilization, pika_messaging)

    return [start_time, total_task_loss, maximum_memory_utilization]

def character_identification_visualization(live, outputs, images, labels, config):
    with torch.no_grad():
        if live:
            output_chars_indexes = []
            demo_outputs = outputs
            demo_images = images
            demo_labels = labels
            if live == 1:
                N = 9
                demo_outputs = outputs[-N:]
                demo_images = images[-N:]
                demo_labels = labels[-N:]

            for output in demo_outputs:
                probs, indices = probabilities_table(output, config.charset)
                output_chars_indexes.append(indices[0].item())
                                                                    
            bitmasked_labels = [config.charBitmasks[index*config.characterMaskSkip] for index in demo_labels]
            bitmasked_outputs = [config.charBitmasks[index*config.characterMaskSkip] for index in output_chars_indexes]


            training_analysis = interweave_arrays(bitmasked_labels, demo_images, bitmasked_outputs)
            print(bytemaps_as_ascii(training_analysis, config.width, config.height, console_width = config.width * 27))

def refresh_refreshables(config, training_size, human_images, gradient_images, training_images, demo_images, human_set_size, complexity_level):
    with torch.no_grad():
        human_dataloader = None
        gradient_dataloaders = None
        training_dataloaders = None
        resized_training_images = None
        resized_demo_images = []
        if not human_set_size and human_set_size != 0:
            human_set_size = len(demo_images)
        if len(demo_images) < human_set_size:
            human_set_size = len(demo_images)

        human_set_size = torch.tensor(float(human_set_size), dtype=torch.float16, requires_grad=True, device = device)

        if config.training_gradients:
            gradient_count = complexity_level if complexity_level >= 1 else torch.tensor(1.0, requires_grad = True)
            gradient_images = create_gradient_tensors(training_size, gradient_count*5, device)
            gradient_dataloaders = create_dataloaders(gradient_images, config, image_batch_limit, shuffle=True, transform=transforms.Compose([DistortionChain()]))

        if config.human_in_the_loop:
            resized_demo_images.extend([resize_image(demo_image, training_size) for demo_image in demo_images])
            human_images = create_gradient_tensors(training_size, human_set_size, device) #set of gradient images in the pool
            human_images.extend(generate_test_images(training_size, human_set_size.int().item(), 5, [0.0, 0.25, 0.5, 0.75, 1])) #set of shape images in the pool
            human_images.extend(random.sample(resized_demo_images, human_set_size.int().item())) #trained photo images in the pool
            #single_shapes, shape_metadata= generate_test_images(training_size, human_set_size.int().item(), 1, [0.0, 1.0], return_metadata=True)
            #human_images.extend(single_shapes) #shape location and identification
            categories = ['gradients', 'shapes', 'user images'] #, 'shape identification']
            human_dataloader = getHumanDataset(human_images, None, categories) #shape_metadata, categories)

        if config.training_images:
            if complexity_level > 0:
                training_images = generate_test_images((256, 256),5, 3, [0.0, 1.0])
            if complexity_level > 1:
                training_images.extend(generate_test_images((256, 256),15,5,[0.0,1.0]))
            if complexity_level > 2:
                training_images.extend(generate_test_images((512, 512),10,10,[0.0,0.2,0.4,0.6,0.8,1.0]))
            if complexity_level > 3:
                training_images.extend(generate_test_images((1024, 1024),5,25,[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
            if complexity_level > 4:
                training_images.extend(generate_test_images((2048, 2048),5,25,[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))

            training_image_count = len(training_images)
            training_dataloaders = create_dataloaders(training_images, config, image_batch_limit, shuffle=True, transform=transforms.Compose([DistortionChain()]))
            resized_training_images = [F.interpolate(img, size=training_size, mode='bilinear', align_corners=False).to(device) for img in training_images]

        return human_images, gradient_images, training_images, human_dataloader, gradient_dataloaders, training_dataloaders, resized_training_images

def getHumanDataset(human_images, shapes_metadata, categories):
    dataset = IndexLabeledTensorDataset(human_images, shapes_metadata, categories)

    # Create a DataLoader
    return DataLoader(dataset, batch_size=1, shuffle=True)

class IndexLabeledTensorDataset(Dataset):
    def __init__(self, tensors, shapes_metadata, categories, choice_limit=4, choice_minimum=2):
        """
        Args:
            tensors (list of Tensors): The list of tensor images.
            shapes_metadata (list): Metadata for the shapes.
            categories (list of str): List of categories representing each segment.
            choice_limit (int): Max number of choices for random selection.
            choice_minimum (int): Minimum number of choices required.
        """
        self.tensors = tensors
        self.shapes_metadata = shapes_metadata
        self.categories = categories
        self.choice_limit = choice_limit
        self.choice_minimum = choice_minimum

    def __len__(self):
        # Adding one more segment for the multichoice category
        segment_length = len(self.tensors) // len(self.categories)
        return segment_length * len(self.categories) + segment_length

    def get_random_sample(self, idx, segment_length, total_length, category_offset=None, include_tensors=True):
        num_choices = self.choice_limit - 1
        valid_indices = list(range(total_length))

        if category_offset is not None:
            valid_indices = list(range(category_offset, category_offset + segment_length))
        
        if idx in valid_indices:
            valid_indices.remove(idx)
        
        sampled_indices = random.sample(valid_indices, min(num_choices, len(valid_indices)))


        if include_tensors:
            return_tensors = [self.tensors[i] for i in sampled_indices]
        else:
            return_tensors = None
        return return_tensors,  sampled_indices

    def __getitem__(self, idx):
        segment_length = len(self.tensors) // len(self.categories)
        total_length = len(self.tensors)
        if idx < segment_length * len(self.categories):
            category_index = idx // segment_length
            category_offset = category_index * segment_length
            returnCategory = self.categories[category_index % len(self.categories)]

            if returnCategory == 'shape identification':
                # Special handling for shape identification
                shape_idx = idx % segment_length
                correct_label = implemented_shapes.index(self.shapes_metadata[shape_idx]['shapes'][0]['shape_type'])
                _, possible_labels = self.get_random_sample(correct_label, len(implemented_shapes), len(implemented_shapes), 0)
                possible_labels.append(correct_label)
                returnMetadata = {
                    'correct_label': len(possible_labels)-1,
                    'possible_labels': possible_labels
                }
                return self.tensors[shape_idx], returnMetadata, returnCategory
            else:
                images, indices = self.get_random_sample(idx, segment_length, total_length, category_offset)
                images.append(self.tensors[idx])
                indices.append(len(images)-1)
                returnMetadata = {
                    'possible_labels': indices,
                    'correct_label': len(indices)-1
                }
                return images, returnMetadata, returnCategory
        else:
            # Multichoice category: Cross-category sample generation
            samples_per_category = self.choice_limit // len(self.categories)
            if not samples_per_category:
                samples_per_category = samples_per_category + 1
            images = []
            indices = []
            for i in range(self.choice_limit):
                if len(images) >= self.choice_limit:
                    break
                if i >= len(self.categories):
                    i = random.randint(0, len(self.categories)-1)
                category_offset = i * segment_length
                
                category_images, category_indices = self.get_random_sample(idx, segment_length, total_length, category_offset)
                images.extend(category_images[:samples_per_category])
                indices.extend(category_indices[:samples_per_category])


            images = images[:self.choice_limit]
            indices = indices[:self.choice_limit]

            returnCategory = 'multichoice'
            returnMetadata = {
                'possible_labels': indices,
                'correct_label': 0  # Set appropriately based on your labeling strategy
            }

            return images, returnMetadata, returnCategory
        
        
        # Return the tensor at index `idx` and also return the index itself as the label

def human_training(config, dataloader):
    """
    This function handles the interaction with human participants to obtain their judgments
    on the provided data samples. It then compares these judgments against expected labels or standards.

    Args:
        dataloader (DataLoader): A PyTorch DataLoader providing batches of data for human judgment.

    Returns:
        selected_labels (Tensor): Labels selected by the human, converted into a format comparable to actual labels.
        actual_labels (Tensor): The correct or expected labels for the data.
    """
    batch_human_logits = []
    correct_labels = []

    for data, metadata, category in dataloader:
        # Display data to the human user and collect their input.
        # This could be done via a GUI where images or data samples are displayed,
        # and the user is asked to assign labels or make selections.
        category = category[0]
        # For demonstration, let's assume a function `collect_human_input()` that encapsulates this process:
        human_label_logits = collect_human_input(config, data, metadata, category)  # Human provides labels based on the displayed data
        batch_human_logits.append(human_label_logits)
        
        # Retrieve the correct label from metadata or a predetermined label set
        correct_labels.append(metadata['correct_label'])

    # Convert lists to tensors for loss calculation
    selected_labels_logits_tensor = torch.stack(batch_human_logits, dim=0)
    actual_labels_tensor = torch.tensor(correct_labels, dtype=torch.long, device = device)

    return selected_labels_logits_tensor, actual_labels_tensor

def task_list_from_training_pattern_and_tasks(training_pattern, training_tasks):
    print(f"training_pattern: {training_pattern}")
    return [use_tensor_as_index(task_index, training_tasks) for task_index in training_pattern]

def set_task_flags(met_thresholds, epoch, total_epochs, *args):
    """
    Determines which tasks to include based on loss thresholds and deadlines.

    Args:
        epoch: The current teaching epoch.
        total_epochs: The total number of teaching epochs.
        args: Tuples of (average loss, loss threshold, deadline as a percent of total epochs)

    Returns:
        list: A list containing: 
              * broken_deadline (bool): True if a deadline has been broken.
              * flags (bool array): True if the corresponding task has met criteria for inclusion
    """

    broken_deadline = False 
    task_count = len(args)
    if not met_thresholds:
        met_thresholds = [False] * task_count
    meeting_thresholds = [False] * task_count
    task_flags = [False] * task_count
    for index, (average_loss, loss_threshold, deadline) in enumerate(args):
        # Determine if the task's deadline has not been passed or if it's the first task or has previously met the threshold
        if met_thresholds[index] or ((epoch / float(total_epochs)) < deadline) or (index == 0 and (epoch /float(total_epochs)) < deadline):
            task_flags[index]= False
            # Retrieve the flag of the previous task to check continuity conditions
            previous_flag = task_flags[index-1] if index > 0 else True
            
            if previous_flag == True and (meeting_thresholds[index - 1] or index == 0):
                # Check if the average loss is greater than zero, an unlikely real value and the initialization value
                if loss_threshold > average_loss > 0:
                    met_thresholds[index] = True
                    meeting_thresholds[index] = True
                    print("THRESHOLD MET")
                else:
                    meeting_thresholds[index] = False
                    print("THRESHOLD NOT MET")
                task_flags[index] = True
                
                
        else:
            broken_deadline = True
            break

    return [met_thresholds, broken_deadline] + task_flags

def train_model(config,  live=False, pika_messaging = pika_messaging):
    epochs_list = []
    training_size = (512, 512)
    
    alpha = torch.tensor(1.5, requires_grad=True)
    beta = torch.tensor(1.5, requires_grad=True)
    gamma = torch.tensor(1.5, requires_grad=True)
    xi = torch.tensor(4.0, requires_grad=True)
    num_epochs = config.epochs
    config = initialize_model(config)

    image_batch_limit = config.image_batch_limit
    gradient_images = []
    training_images = []
    demo_images = []
    hitl_images = []

    if config.demo_image:
        demo_images.append(pilToPreASCIITensor(config.demo_image))

    if config.demo_images:
        demo_images.extend(load_images_as_tensors(device))    

    demo_image_count = len(demo_images)
    demo_dataloaders = create_dataloaders(demo_images, config, image_batch_limit, shuffle=True, transform=transforms.Compose([DistortionChain()]))
    resized_demo_images = [F.interpolate(img, size=training_size, mode='bilinear', align_corners=False).to(device) for img in demo_images]
    
    
    maximum_memory_utilization = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    


    training_task_names = []
    if config.training_categories:
        training_task_names.append('identification')
    if config.training_gradients:
        training_task_names.append('gradients')
    if config.training_images:
        training_task_names.append('shapes')
    if config.demo_images or config.demo_image:
        training_task_names.append('user images')
    if config.human_in_the_loop:
        training_task_names.append('human in the loop')
    
    
    hitl_images, gradient_images, training_images, human_dataloader, gradient_dataloaders, training_dataloaders, resized_training_images = refresh_refreshables(config, training_size, hitl_images, gradient_images, training_images, demo_images, 5, config.complexity_level)


    
    outer_learning = False
    num_teaching_epochs = 0
    num_outer_epochs = 100
    if override_meta_model_learning:
        num_teaching_epochs = 1000
    else:
        num_teaching_epochs = 1000

    if not config.metamodel:
        config.metamodel = MetaModel()
    meta_weight_decay = .0000001
    config.metamodel.train()
    meta_optimizer = optim.AdamW(config.metamodel.parameters(), lr=0.0001, weight_decay=meta_weight_decay)
    training_image_loss_function = demo_loss_function = gradient_loss_function = human_loss_function = categorization_loss_function = None
    training_image_optimizer = demo_optimizer = gradient_optimizer = categorization_optimizer = human_optimizer = optimizer = None    
    
    for outer_epoch in range(num_outer_epochs):
        current_total_student_epochs = 0

        average_category_loss = torch.tensor(0.0, device=device, requires_grad=True)
        average_gradient_loss = torch.tensor(0.0, device=device, requires_grad=True)
        average_image_loss = torch.tensor(0.0, device=device, requires_grad=True)
        average_demo_loss = torch.tensor(0.0, device=device, requires_grad=True)
        average_human_loss = torch.tensor(0.0, device=device, requires_grad=True)
        average_total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        task_flag_data = [(average_category_loss, 6.5, .01), (average_human_loss, 6.5, .02), (average_gradient_loss, 6.5, .05), (average_image_loss, 6.5, .1), (average_demo_loss, 6.5, .2)]
        flags = set_task_flags(None, 0, num_teaching_epochs, *task_flag_data)
        met_thresholds, broken_deadline, config.training_categories, config.human_in_the_loop, config.training_gradients, config.training_images, config.demo_images = flags
        
        previous_category_loss = torch.tensor(0.0, device=device, requires_grad=True)
        previous_gradient_loss = torch.tensor(0.0, device=device, requires_grad=True)
        previous_image_loss = torch.tensor(0.0, device=device, requires_grad=True)
        previous_demo_loss = torch.tensor(0.0, device=device, requires_grad=True)
        previous_human_loss = torch.tensor(0.0, device=device, requires_grad=True)
        previous_average_loss = torch.tensor(0.0, device=device, requires_grad=True)

        student_category_losses = []
        student_gradient_losses = []
        student_image_losses = []
        student_demo_losses = []
        student_human_losses = []
        student_average_losses = []

        categorization_losses = []
        gradient_losses = []
        image_similarity_losses = []
        demo_image_similarity_losses = []
        human_losses = []
        persistence = 1
        teaching_offset = 0
        if not persistence or outer_epoch == 0:
            config.metamodel.reset_sequence()
            teaching_offset = 1
            if config.model_loaded == 0:
                config = initialize_model(config, force_reset=True)
                teaching_offset = 0
            config.model.train()
            training_image_loss_function = get_loss_function(config.training_loss_function)
            demo_loss_function = get_loss_function(config.demo_loss_function)
            gradient_loss_function = get_loss_function(config.gradient_loss_function)
            human_loss_function = nn.CrossEntropyLoss()
            categorization_loss_function = nn.CrossEntropyLoss()
            weight_decay = .000005
            training_image_optimizer = optim.AdamW(config.model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
            demo_optimizer = optim.AdamW(config.model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
            gradient_optimizer = optim.AdamW(config.model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
            categorization_optimizer = optim.AdamW(config.model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
            human_optimizer = optim.AdamW(config.model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
            
            optimizer = optim.AdamW(config.model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
        for teaching_epoch in range(teaching_offset, num_teaching_epochs + teaching_offset):
            training_metrics = torch.tensor([
    current_total_student_epochs, average_category_loss, average_gradient_loss, average_image_loss, 
    average_demo_loss, average_human_loss, previous_category_loss, previous_gradient_loss, previous_image_loss, 
    previous_demo_loss, previous_human_loss, average_total_loss, previous_average_loss, config.complexity_level, maximum_memory_utilization
], dtype=torch.float32)

            meta_model_output = config.metamodel(training_metrics)

            task_flag_data = [(average_category_loss, 6.5, .01), (average_human_loss, 6.5, .02), (average_gradient_loss, 6.5, .05), (average_image_loss, 6.5, .1), (average_demo_loss, 6.5, .2)]
            flags = set_task_flags(met_thresholds, teaching_epoch, num_teaching_epochs, *task_flag_data)
            met_thresholds, broken_deadline, config.training_categories, config.human_in_the_loop, config.training_gradients, config.training_images, config.demo_images = flags
            if not outer_epoch and not teaching_epoch:
                config.training_categories = config.training_gradients = config.training_images = config.demo_images = config.human_in_the_loop = True
            if broken_deadline:
                break

            training_task_names = []
            if config.training_categories:
                training_task_names.append('identification')
            if config.training_gradients:
                training_task_names.append('gradients')
            if config.training_images:
                training_task_names.append('shapes')
            if config.demo_images or config.demo_image:
                training_task_names.append('user images')
            if config.human_in_the_loop:
                training_task_names.append('human in the loop')

            training_pattern, config.complexity_level, outer_learning, num_epochs, num_category_epochs, num_gradient_epochs, num_image_epochs, num_demo_epochs, num_human_epochs, outer_learning_rate, category_learning_rate, gradient_learning_rate, image_learning_rate, demo_learning_rate, human_learning_rate = config.metamodel.interpret_output(meta_model_output, training_task_names, demo_image_count=demo_image_count)
            
            if not outer_epoch and not teaching_epoch:
                outer_learning = torch.tensor(1.0, requires_grad=True)
            
            config = initialize_model(config, complexity_level=config.complexity_level, force_datamodel_only=True)
            maximum_memory_utilization = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
            config.characterMaskSkip = (len(config.charBitmasks)//len(config.charset))
            print(f"Complexity: {config.complexity_level}, Outer learning: {outer_learning}, Number of epochs: {num_epochs+1}, Category epochs: {num_category_epochs+1}, Gradient epochs: {num_gradient_epochs+1}, Image epochs: {num_image_epochs+1}, Demo epochs: {num_demo_epochs+1}, Human learning rate: {human_learning_rate}, Outer learning rate: {outer_learning_rate}")
            print(f"Teaching Epoch {teaching_epoch + 1}/{num_teaching_epochs} Learning Rates: Category: {category_learning_rate}, Gradient: {gradient_learning_rate}, Image: {image_learning_rate}, Demo: {demo_learning_rate}, Human: {human_learning_rate}")
            average_category_loss = torch.tensor(0.0, device=device, requires_grad=True)
            average_gradient_loss = torch.tensor(0.0, device=device, requires_grad=True)
            average_image_loss = torch.tensor(0.0, device=device, requires_grad=True)
            average_demo_loss = torch.tensor(0.0, device=device, requires_grad=True)
            average_human_loss = torch.tensor(0.0, device=device, requires_grad=True)
            average_total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            start_time = time.time() 
            for epoch in range(torch.round(num_epochs).int()+1):
                current_memory_utilization = get_gpu_memory_utilization(device_id)
                print(f"Current Memory Utilization: {current_memory_utilization}")
                if current_memory_utilization > 75:
                    break
                adjust_learning_rate(optimizer, outer_learning_rate)
                previous_category_loss = torch.tensor(0.0, device=device, requires_grad=True)
                previous_gradient_loss = torch.tensor(0.0, device=device, requires_grad=True)
                previous_image_loss = torch.tensor(0.0, device=device, requires_grad=True)
                previous_demo_loss = torch.tensor(0.0, device=device, requires_grad=True)
                previous_average_loss = torch.tensor(0.0, device=device, requires_grad=True)

                current_total_student_epochs += 1
                config.epoch = epoch

                

                if not skip_refreshments:
                    refreshStartTime = time.time()
                    hitl_images, gradient_images, training_images, human_dataloader, gradient_dataloaders, training_dataloaders, resized_training_images = refresh_refreshables(config, training_size, hitl_images, gradient_images, training_images, demo_images, 5, config.complexity_level)
                    refreshEndTime = time.time()
                    refreshDuration = refreshEndTime - refreshStartTime
                    start_time = start_time + refreshDuration

                
                total_categorization_loss = torch.tensor(0.0, device=device, requires_grad=True)
                total_gradient_image_similarity_loss = torch.tensor(0.0, device=device, requires_grad=True)
                total_image_similarity_loss = torch.tensor(0.0, device=device, requires_grad=True)
                total_demo_image_similarity_loss = torch.tensor(0.0, device=device, requires_grad=True)
                total_human_loss = torch.tensor(0.0, device=device, requires_grad=True)


                humaninthelooptask = {
                    'human in the loop': {
                        'optimizer': human_optimizer,
                        'learning_rate': human_learning_rate,
                        'num_epochs' : num_human_epochs,
                        'dataloaders': [human_dataloader],
                        'loss_function' : human_loss_function,
                        'loss_alpha' : xi,
                        'original_images' : hitl_images
                    }}
                categorizingtask = {
                    'identification': {
                        'optimizer': categorization_optimizer,
                        'learning_rate': category_learning_rate,
                        'num_epochs': num_category_epochs,
                        'dataloaders': [config.dataloader],
                        'loss_function': categorization_loss_function,
                        'loss_alpha': torch.tensor(1.0, requires_grad=True),
                        'original_images': None
                    }}
                gradientimagetask = {
                    'gradients': {
                        'optimizer': gradient_optimizer,
                        'learning_rate': gradient_learning_rate,
                        'num_epochs': num_gradient_epochs,
                        'dataloaders': gradient_dataloaders,
                        'loss_function': gradient_loss_function,
                        'loss_alpha': alpha,
                        'original_images': gradient_images
                    }}
                testimagetask = {
                    'shapes': {
                        'optimizer': training_image_optimizer,
                        'learning_rate': image_learning_rate,
                        'num_epochs': num_image_epochs,
                        'dataloaders': training_dataloaders,
                        'loss_function': training_image_loss_function,
                        'loss_alpha': beta,
                        'original_images': resized_training_images
                    }}
                demoimagetask = {
                    'user images': {
                        'optimizer': demo_optimizer,
                        'learning_rate': demo_learning_rate,
                        'num_epochs': num_demo_epochs,
                        'dataloaders': demo_dataloaders,
                        'loss_function': demo_loss_function,
                        'loss_alpha': gamma,
                        'original_images': resized_demo_images
                    }}
                training_tasks = []
                # List of task dictionaries
                if config.training_categories:
                    training_tasks.append(categorizingtask)
                if config.training_gradients:
                    training_tasks.append(gradientimagetask)
                if config.training_images:
                    training_tasks.append(testimagetask)
                if config.demo_images or config.demo_image:
                    training_tasks.append(demoimagetask)
                if config.human_in_the_loop:
                    training_tasks.append(humaninthelooptask)

                #training_tasks = [categorizingtask, gradientimagetask, testimagetask, demoimagetask, humaninthelooptask]
                task_list = task_list_from_training_pattern_and_tasks(training_pattern, training_tasks)

                # Define an empty dictionary to hold all tasks
                tasks = {}
                memory_thresholds = []
                # Update the tasks dictionary with the randomized task entries
                for task_dict in task_list:
                    tasks.update(task_dict)

                allowances = 90 / (len(tasks)+1)
                for i in range(len(task_list)):
                    memory_thresholds.append(90 - allowances * i)
                turns = {}
                
                for index, (task_name, params) in enumerate(tasks.items()):
                    start_time, total_task_loss, maximum_memory_utilization = run_training_loop(
                        maximum_memory_utilization=maximum_memory_utilization,
                        task_type=task_name,
                        optimizer=params['optimizer'],
                        learning_rate=params['learning_rate'],
                        num_epochs=params['num_epochs'],
                        dataloaders=params['dataloaders'],
                        loss_function=params['loss_function'],
                        loss_alpha=params['loss_alpha'],
                        original_images=params['original_images'],
                        outer_learning=outer_learning,
                        config=config,
                        live=live,
                        training_size=training_size,
                        start_time=start_time,
                        outer_epoch=outer_epoch,
                        teaching_epoch=teaching_epoch,
                        epoch=epoch, 
                        pika_messaging=pika_messaging,
                        memory_threshold = memory_thresholds[index]  
                    )
                    if not task_name in turns:
                        turns[task_name] = 0
                    turns[task_name] = turns[task_name] + 1
                    if task_name == 'gradients':    
                        total_gradient_image_similarity_loss = total_gradient_image_similarity_loss + total_task_loss
                    elif task_name == 'shapes':
                        total_image_similarity_loss = total_image_similarity_loss + total_task_loss
                    elif task_name == 'user images':
                        total_demo_image_similarity_loss = total_demo_image_similarity_loss + total_task_loss
                    elif task_name == 'identification':
                        total_categorization_loss = total_categorization_loss + total_task_loss
                    elif task_name == 'human in the loop':
                        total_human_loss = total_human_loss + total_task_loss


                if 'gradients' in turns:
                    total_gradient_image_similarity_loss = total_gradient_image_similarity_loss / turns['gradients']
                    gradient_losses.append(total_gradient_image_similarity_loss)
                if 'shapes' in turns:
                    total_image_similarity_loss = total_image_similarity_loss / turns['shapes']
                    image_similarity_losses.append(total_image_similarity_loss)
                if 'user images' in turns:
                    total_demo_image_similarity_loss = total_demo_image_similarity_loss / turns['user images']
                    demo_image_similarity_losses.append(total_demo_image_similarity_loss)
                if 'identification' in turns:
                    total_categorization_loss = total_categorization_loss / turns['identification']
                    categorization_losses.append(total_categorization_loss)
                if 'human in the loop' in turns:
                    total_human_loss = total_human_loss / turns['human in the loop']
                    human_losses.append(total_human_loss)

                previous_category_loss = categorization_losses[-1] if 'identification' in turns else torch.tensor(0.0, requires_grad=True)
                previous_gradient_loss = gradient_losses[-1] if 'gradients' in turns else torch.tensor(0.0, requires_grad=True)
                previous_image_loss = image_similarity_losses[-1] if 'shapes' in turns else torch.tensor(0.0, requires_grad=True)
                previous_demo_loss = demo_image_similarity_losses[-1] if 'user_images' in turns else torch.tensor(0.0, requires_grad=True)
                previous_human_loss = human_losses[-1] if 'human in the loop' in turns else torch.tensor(0.0, requires_grad=True)
                previous_losses = [previous_category_loss, previous_gradient_loss, previous_image_loss, previous_demo_loss, previous_human_loss]
                number_of_active_losses = 0
                for loss in previous_losses:
                    if loss != 0:
                        number_of_active_losses = number_of_active_losses + 1 
                previous_average_loss = (previous_category_loss + previous_gradient_loss + previous_image_loss + previous_demo_loss + previous_human_loss)/number_of_active_losses
                

                if outer_learning:
                    optimizer.zero_grad()
                    previous_average_loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
                if pika_messaging:
                    pika_start = time.time()
                    message_data = {
                        'outer_epoch': outer_epoch+1,
                        'teaching_epoch': teaching_epoch+1,
                        'homework_epoch': epoch+1,
                        'outer_learning': outer_learning.item(),
                        'learning_rate': outer_learning_rate,
                        'batch_total_loss': previous_average_loss.item()
                    }
                    message = json.dumps(message_data)
                    channel.basic_publish(exchange='', routing_key='ascii_statistics_queue', body=message)
                    pika_end = time.time()
                    pika_time = pika_end - pika_start
                    start_time = start_time + pika_time                

                epochs_list.append(epoch + 1)

                model_save_path = f"{config.model_path}_epoch_{epoch+1}.pth"
  
                average_loss = torch.tensor(0.0, requires_grad = True, device=device)
                if len(categorization_losses) and config.training_categories:
                    print(f"Epoch {epoch + 1}, Total Categorization Loss: {categorization_losses[-1].item()}")
                    average_loss = average_loss + categorization_losses[-1].item()
                if len(gradient_losses) and config.training_gradients:
                    print(f"Epoch {epoch + 1}, Total Gradient Image Similarity Loss: {gradient_losses[-1].item()}")
                    average_loss = average_loss + gradient_losses[-1].item()
                if len(image_similarity_losses) and config.training_images:
                    print(f"Epoch {epoch + 1}, Total Image Similarity Loss: {image_similarity_losses[-1].item()}")
                    average_loss = average_loss + image_similarity_losses[-1].item()
                if len(demo_image_similarity_losses) and (config.demo_image or config.demo_images):
                    print(f"Epoch {epoch + 1}, Total Demo Image Similarity Loss: {demo_image_similarity_losses[-1].item()}")
                    average_loss = average_loss + demo_image_similarity_losses[-1].item()
                if len(human_losses) and config.human_in_the_loop:
                    print(f"Epoch {epoch + 1}, Total Human Validator Loss: {human_losses[-1].item()}")
                    average_loss = average_loss + human_losses[-1].item()
                
                average_loss = average_loss / len(turns.keys())
                print(f"Epoch [{epoch + 1}/{num_epochs+1}], Combined Average Loss: {average_loss}")

            student_category_losses.append(previous_category_loss)
            student_gradient_losses.append(previous_gradient_loss)
            student_image_losses.append(previous_image_loss)
            student_demo_losses.append(previous_demo_loss)
            student_human_losses.append(previous_human_loss)
            student_average_losses.append(previous_average_loss)

            average_category_loss = sum(student_category_losses) / max(sum(1 for loss in student_category_losses if loss != 0), 1)
            average_gradient_loss = sum(student_gradient_losses) / max(sum(1 for loss in student_gradient_losses if loss != 0), 1)
            average_image_loss = sum(student_image_losses) / max(sum(1 for loss in student_image_losses if loss != 0), 1)
            average_demo_loss = sum(student_demo_losses) / max(sum(1 for loss in student_demo_losses if loss != 0), 1)
            average_human_loss = sum(student_human_losses) / max(sum(1 for loss in student_human_losses if loss != 0), 1)
            average_total_loss = sum(student_average_losses) / max(sum(1 for loss in student_average_losses if loss != 0), 1)
            
            end_time = time.time()
       
            epoch_duration = end_time - start_time  
            if pika_messaging:
                message_data = {
                    'outer_epoch': outer_epoch+1,
                    'teaching_epoch': teaching_epoch+1,
                    'teaching_epoch_time': epoch_duration,
                    'previous_category_loss': previous_category_loss.item(),
                    'previous_gradient_loss': previous_gradient_loss.item(),
                    'previous_image_loss': previous_image_loss.item(),
                    'previous_demo_loss': previous_demo_loss.item(),
                    'previous_human_loss': previous_human_loss.item(),
                    'previous_average_loss': previous_average_loss.item(),
                    'average_category_loss': average_category_loss.item(),
                    'average_gradient_loss': average_gradient_loss.item(),
                    'average_image_loss': average_image_loss.item(),
                    'average_demo_loss': average_demo_loss.item(),
                    'average_human_loss': average_human_loss.item(),
                    'average_total_loss': average_total_loss.item()
                }
                message = json.dumps(message_data)
                            
                channel.basic_publish(exchange='', routing_key='ascii_statistics_queue', body=message)
            if not override_meta_model_learning:           
                if len(student_average_losses) > 1:
                    loss_reduction_per_second = torch.tensor((student_average_losses[-2].item() - student_average_losses[-1]).item(), requires_grad=True) / epoch_duration
                else:
                    loss_reduction_per_second = torch.tensor(0.0, requires_grad=True)

            
                if len(student_average_losses) > 2:
                    student_average_losses[-3] = student_average_losses[-3].detach().to('cpu')
                meta_loss = torch.clamp((1 - loss_reduction_per_second),.00001,100) 

                # Get GPU utilization and calculate penalty
                
                mem_penalty = memory_penalty(maximum_memory_utilization)
                # Apply the memory penalty to meta_loss
                meta_loss = meta_loss + mem_penalty

                print(f"pass number: {teaching_epoch}, meta_loss: {meta_loss.item()}")
                print(f"End of Teaching Epoch {teaching_epoch + 1}, Losses Stored: Category - {student_category_losses[-1].item()}, Gradient - {student_gradient_losses[-1].item()}, Image - {student_image_losses[-1].item()}, Demo - {student_demo_losses[-1].item()}, Human - {student_human_losses[-1].item()}")
                print(f"Teaching Epoch {teaching_epoch + 1}, Meta Loss: {meta_loss.item()}, Improvement Rate: {loss_reduction_per_second.item()}")

                # Backward pass for the MetaModel

                
                meta_loss.backward()
                torch.cuda.empty_cache()
            meta_optimizer.step()
            meta_optimizer.zero_grad()
            if not quick_test:
                save_model(None, meta_model=config.metamodel)
                save_model(config)
            torch.cuda.empty_cache()
        
def memory_penalty(mem_utilization, tipping_point=50, start_point=65, steep_point=80):
    returnVal = 0
    if mem_utilization < start_point and mem_utilization > tipping_point:
        return 0
    elif mem_utilization <= tipping_point:
        return (tipping_point - mem_utilization)/100
    elif mem_utilization >= start_point:
        returnVal = (mem_utilization-start_point)/100
    if mem_utilization >= steep_point:
        return (mem_utilization - steep_point)/100 + returnVal
    return returnVal
    

def list_printable_characters(font_path, font_size=12, epsilon=1):
    font = TTFont(font_path)
    cmap = font.getBestCmap()

    # Define a list of Unicode ranges covering a broad spectrum of scripts
    # This list should be expanded based on specific needs and common usage
    unicode_ranges = [
        range(0x0020, 0x007E),  # Basic Latin
        range(0x00A0, 0x00FF),  # Latin-1 Supplement
        range(0x0100, 0x017F),  # Latin Extended-A
        range(0x0370, 0x03FF),  # Greek and Coptic
        range(0x0400, 0x04FF),  # Cyrillic
        range(0x0590, 0x05FF),  # Hebrew
        range(0x0600, 0x06FF),  # Arabic
        range(0x0900, 0x097F),  # Devanagari
        range(0x4E00, 0x9FFF),  # CJK Unified Ideographs
        range(0x0E00, 0x0E7F),  # Thai
        range(0x10A0, 0x10FF),  # Georgian
        range(0x1D00, 0x1D7F),  # Phonetic Extensions
        range(0x2000, 0x206F),  # General Punctuation
        range(0x20A0, 0x20CF),  # Currency Symbols
        range(0x2100, 0x214F)   # Letterlike Symbols
    ]

    # Filter characters based on the font's cmap and the defined safe ranges
    safe_printable_chars = [chr(code) for r in unicode_ranges for code in r if code in cmap]

    # Calculate size statistics for each character
    font_sizes = []
    font = ImageFont.truetype(font_path, font_size)
    char_images = {}
    for char in safe_printable_chars:
        fontsize, fontplacement = font.font.getsize(char)
        img = Image.new('L', fontsize, color=255)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=font, fill=0)
        char_images[char] = np.array(img)

    # Remove duplicates
    unique_chars = []
    unique_font_sizes = []
    seen_images = set()
    for char, img in char_images.items():
        img_tuple = tuple(map(tuple, img))
        if img_tuple not in seen_images:
            seen_images.add(img_tuple)
            unique_chars.append(char)
            (width, height), _ = font.font.getsize(char)
            unique_font_sizes.append(width)

    # Remove outliers to maintain a width variance of some epsilon
    mode_width = max(set(unique_font_sizes), key=unique_font_sizes.count)
    unique_chars = [char for char, width in zip(unique_chars, unique_font_sizes) if abs(width - mode_width) <= epsilon]
    unique_font_sizes = [width for width in unique_font_sizes if abs(width - mode_width) <= epsilon]
    return ''.join(unique_chars)
    #return ''.join(unique_chars), unique_font_sizes


def generate_checkerboard_pattern(width, height, block_size=2):
    pattern = np.zeros((height, width), dtype=np.uint8)
    # Use 191 for 75% brightness (75% of 255 is roughly 191)
    # Use 64 for 25% brightness (25% of 255 is roughly 64)
    light_gray = 191
    dark_gray = 64
    
    for y in range(0, height, block_size * 2):
        for x in range(0, width, block_size * 2):
            # Light gray blocks
            pattern[y:y + block_size, x:x + block_size] = light_gray
            pattern[y + block_size:y + block_size * 2, x + block_size:x + block_size * 2] = light_gray
            # Dark gray blocks
            pattern[y + block_size:y + block_size * 2, x:x + block_size] = dark_gray
            pattern[y:y + block_size, x + block_size:x + block_size * 2] = dark_gray

    return pattern


def generate_variants(charset, fonts, max_width, max_height, level):
    charBitmasks = []
    background_colors = [0]
    text_colors = [255]

    if level >= 2:
        background_colors += [ 255 ]
        text_colors += [ 0 ]
    if level >= 3:
        background_colors += [int(255 * 0.75), int(255 * 0.25)]
        text_colors += [int(255 * 0.25), int(255 * 0.75)]

    if level >= 4:
        background_colors += [0, int(255 * 0.95)]
        text_colors += [int(255 * 0.05), 255]
    
    for char in charset:
        for font in fonts:
            # Measure text size to calculate proper alignment
            (width, height), (offset_x, offset_y) = font.font.getsize(char)
            text_width, text_height = width, height + offset_y

            # Calculate the position to center the text in the image
            x_position = (max_width - text_width) // 2
            y_position = (max_height - text_height) // 2

            for bg_color, text_color in zip(background_colors, text_colors):
                image = Image.new('L', (max_width, max_height), color=bg_color)
                draw = ImageDraw.Draw(image)
                # Draw text centered in the image
                draw.text((x_position, y_position), char, font=font, fill=text_color)
                bitmask = np.array(image)
                charBitmasks.append(bitmask)
            
            if level >= 5:
                checkerboard = generate_checkerboard_pattern(max_width, max_height)
                image = Image.fromarray(checkerboard).convert('L')
                draw = ImageDraw.Draw(image)
                # Also center text in checkerboard pattern
                draw.text((x_position, y_position), char, font=font, fill=255)
                bitmask = np.array(image)
                charBitmasks.append(bitmask)
    
    return charBitmasks

def bytemaps_as_ascii(charBitmasks, width, height, console_width=240, ascii_gradient=" .:-=+*#%@"):
    num_bytemaps_per_line = console_width // width
    lines_per_bytemap = height
    lines_to_print = ""

    for i in range(0, len(charBitmasks), num_bytemaps_per_line):
        bytemap_row = charBitmasks[i:i + num_bytemaps_per_line]
        for line_num in range(lines_per_bytemap):
            for bitmask in bytemap_row:
                # Convert to numpy if it's a tensor and ensure correct dimensionality
                if isinstance(bitmask, torch.Tensor):
                    bitmask = bitmask.squeeze().cpu().numpy()  # Squeeze to remove single-dimensional entries

                # Normalize the bytemap if necessary
                if bitmask.max() > 1:
                    bitmask = bitmask / 255.0

                for j in range(width):
                    value = bitmask[line_num, j]  # Get the intensity value at the current pixel
                    char_index = int(value * (len(ascii_gradient) - 1))
                    lines_to_print += ascii_gradient[char_index]
                lines_to_print += " "
            lines_to_print += "\n"
        lines_to_print += "\n"

    return lines_to_print





def obtain_charset(font_files=None, font_size=None, complexity_level=0, config=None):
    global charset
    if config and font_files == None and font_size == None:
        font_files = config.font_files
        font_size = config.font_size

    if charset == "":
        charset = list_printable_characters(font_files[0], font_size)
    max_width = max_height = 0
    fonts = []
    for font_path in font_files:
        font = ImageFont.truetype(font_path, font_size)
        for char in charset:
            (width, height), (offset_x, offset_y) = font.font.getsize(char)
            max_width = max(max_width, width)
            max_height = max(max_height, height + offset_y)
        fonts.append(font)
    
    charBitmasks = generate_variants(charset, fonts, max_width, max_height, complexity_level)
    
    return fonts, charset, charBitmasks, max_width, max_height

defaultReferenceFonts, defaultCharset, defaultCharBitmasks, defaultWidth, defaultHeight = obtain_charset(font_files, font_size, COMPLEXITY_LEVEL)
if live:
    print(bytemaps_as_ascii(defaultCharBitmasks, defaultWidth, defaultHeight))
default_compat_config = ModelCompatConfig(defaultCharset, font_files, font_size, conv1_out, conv2_out, linear_out, defaultWidth, defaultHeight)
default_config = ModelConfig(default_compat_config, dropout, learning_rate, epochs, batch_size, demo_image, epochs_per_preview)

def create_config(base_config, font_files=None, font_size=None, complexity_level=0):
    if not font_files:
        font_files = base_config.font_files
    if not font_size:
        font_size = base_config.font_size
    
    referenceFont, charset, char_bitmasks, width, height = obtain_charset(font_files, font_size, complexity_level)
    compat_config = ModelCompatConfig(
        charset=charset,
        font_files=font_files,
        font_size=font_size,
        conv1_out=base_config.conv1_out,
        conv2_out=base_config.conv2_out,
        linear_out=base_config.linear_out,
        width=width,
        height=height
    )
    return ModelConfig(
        compatibility_model=compat_config,
        dropout=base_config.dropout,
        learning_rate=base_config.learning_rate,
        epochs=base_config.epochs,
        batch_size=base_config.batch_size,
        demo_image=base_config.demo_image
    )


          
    



    

def load_model(config, path=None, device=torch.device('cpu'), meta_model=None):
    """Loads a model from the specified path."""
    if meta_model:
        meta_model = MetaModel()
        path = metamodel_path
        if os.path.exists(path):
            meta_model.load_state_dict(torch.load(path))
            print("Metamodel loaded")
        else:
            print("Metamodel not found")
        return meta_model
    else:
        if path is None:
            path = os.path.join(some_model_directory, f"{config.model_path}.pth")

        if os.path.exists(path):
            config = initialize_model(config)
            config.model.load_state_dict(torch.load(path, map_location=device))
            print("Model loaded successfully.")
            return True
        else:
            print("Model file does not exist.")
            return False

class CustomSimpleShapesDataset(Dataset):
    def __init__(self, size, length, config, complexity, transform=None):
        self.test_images = generate_test_images(size, length, complexity)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.test_images[idx]
        return image      

    def __len__(self):
        return len(self.test_images)

def obtain_training_dataset(config, complexity_level):
    referenceFont, config.charset, config.charBitmasks, config.width, config.height = obtain_charset(config=config, complexity_level=complexity_level)
    X_training = np.array(config.charBitmasks)
    X_training = X_training.reshape(len(config.charBitmasks), -1).astype(np.float32)
    label_repeats = len(config.charBitmasks) // len(config.charset)
    Y_training = np.repeat(np.arange(len(config.charset)),label_repeats)

    X_tensor = torch.tensor(X_training)
    Y_tensor = torch.tensor(Y_training, dtype=torch.long)
    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    transform = transforms.Compose([
        ToTensorAndToDevice(device=device),
        DistortionChain(),
    ])

    return CustomDataset(config.charBitmasks, Y_tensor, transform=transform), config.charBitmasks, config.width, config.height

def obtain_model(config, input_tensor_images=None, force_reset=False, complexity_level=0):
    if input_tensor_images == None:
        dataset, config.charBitmasks, config.width, config.height = obtain_training_dataset(config, complexity_level)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    else:
        referenceFont, config.charset, config.charBitmasks, config.width, config.height = obtain_charset(config, complexity_level)
        dataset = obtain_custom_input_dataset(input_tensor_images)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    if not config.model or force_reset:
        model = CharSorter(config)
        model = model.to(device)
    else:
        model = config.model
    
    return model, dataloader

def initialize_model(config, input_tensor_images=None, force_reset=False, complexity_level=0, force_datamodel_only=False):
    if force_datamodel_only:
        config.model, config.dataloader = obtain_model(config, input_tensor_images, False, complexity_level)
    if not (config.model and config.dataloader) or force_reset:
        config.model, config.dataloader = obtain_model(config, input_tensor_images, force_reset, complexity_level)    
    config.complexity_level = complexity_level
    return config

def render_text_lines(charBitmasks, text, charset, max_width, max_height):
    lines = text.split('\n')
    char_tensors = [torch.from_numpy(bitmask) for bitmask in charBitmasks]

    num_lines = len(lines)
    line_length = max(len(line) for line in lines)
    variants_per_char = len(charBitmasks) // len(charset)  # Ensure this uses integer division
    text_tensor = torch.zeros(num_lines, max_height, line_length * max_width, dtype=torch.uint8).to(device)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char in charset:
                index = charset.index(char) * variants_per_char  # Get the first variant (assuming it's white on black)
                text_tensor[i, :, j * max_width:(j + 1) * max_width] = char_tensors[int(index)]
    
    return text_tensor

def text_tensor_to_image_tensor(tensor):
    if tensor.dim() != 3:
        raise ValueError("Tensor must be 3D for multi-line text images.")
    tensor = tensor.view(-1, tensor.size(2))  # Flatten all lines vertically
    return tensor

def tensor_to_image(tensor):
    tensor = text_tensor_to_image_tensor(tensor)
    np_array = tensor.cpu().numpy()
    return Image.fromarray(np_array, 'L')
def get_loss_function(type):
    if type == 'mse':
        return torch.nn.MSELoss()
    elif type == 'mae':
        return torch.nn.L1Loss()
    elif type == 'psnr':
        return PSNRLoss()
    else:
        raise ValueError('Invalid loss function type')

class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, input, target):
        mse = torch.mean((input - target) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        return psnr
def create_gradient_tensor(size, device, gradient_type='linear', p1=(0, 0), p2=(1, 1)):
    height, width = size
    
    if gradient_type == 'linear':
        x = torch.linspace(0, 1, steps=width)
        y = torch.linspace(0, 1, steps=height)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
        
        # Calculate the direction vector from p1 to p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate the projection of each point onto the line p1-p2
        projections = ((x_grid - p1[0]) * dx + (y_grid - p1[1]) * dy) / (dx**2 + dy**2)**0.5
        
        # Create a linear gradient from 0 at p1 to 1 at p2
        gradient = projections / projections.max()
    
    elif gradient_type == 'radial':
        x = torch.linspace(0, 1, steps=width)
        y = torch.linspace(0, 1, steps=height)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
        
        # Calculate the distance from p1 to each point in the grid
        distances = ((x_grid - p1[0])**2 + (y_grid - p1[1])**2)**0.5
        
        # Create a radial gradient from 0 at p1 to 1 at the maximum distance
        gradient = distances / distances.max()
    
    return gradient.to(device).unsqueeze(0).unsqueeze(0)

def blend_tensors(tensor1, tensor2, operation):
    if operation == 'add':
        return torch.clamp(tensor1 + tensor2, 0, 1)
    elif operation == 'multiply':
        return tensor1 * tensor2
    elif operation == 'difference':
        return torch.abs(tensor1 - tensor2)

def generate_tensor(size, device):
    while True:
        tensors = []
        for _ in range(3):
            gradient_type = random.choice(['linear', 'radial'])
            if gradient_type == 'linear':
                p1 = (random.random(), random.random())
                p2 = (random.random(), random.random())
                tensor = create_gradient_tensor(size, device, gradient_type, p1, p2)
            else:
                p1 = (random.random(), random.random())
                tensor = create_gradient_tensor(size, device, gradient_type, p1)
            
            tensors.append(tensor)
        
        operations = [random.choice(['add', 'multiply', 'difference']) for _ in range(2)]
        
        result = blend_tensors(blend_tensors(tensors[0], tensors[1], operations[0]), tensors[2], operations[1])
        
        # Normalize the resulting tensor
        result = result / result.max()
        
        # Check if the resulting tensor is acceptable
        low_values = (result < 0.25).sum() / (size[0] * size[1])
        high_values = (result > 0.25).sum() / (size[0] * size[1])
        
        if low_values > .1 and high_values > .1:
            return result
            
def create_gradient_tensors(size, count, device):
    returnTensors = []
    for _ in range(torch.round(count).int()):
        returnTensors.append(generate_tensor(size, device))
    
    return returnTensors
def interweave_arrays(*arrays):
    # Find the maximum length of any array to handle uneven lengths
    max_length = max(len(arr) for arr in arrays)
    interwoven = []
    for i in range(max_length):
        for array in arrays:
            if i < len(array):  # Check if the index is within the current array's length
                interwoven.append(array[i])
    return interwoven
from torch.utils.data import DataLoader, Subset

def create_dataloaders(image_list, config, pixel_limit, shuffle=True, transform=None):
    dataloaders = []
    current_batch_indices = []  # To store indices of images in the current batch
    current_pixel_count = 0

    for index, image in enumerate(image_list):
        # Calculate the number of pixels in the image tensor
        num_pixels = image.numel()  # Assuming image is already a tensor

        if current_pixel_count + num_pixels > pixel_limit:
            # If adding this image would exceed the limit, finalize the current batch
            if current_batch_indices:
                subset = Subset(image_list, current_batch_indices)
                dataloader = obtainDataloaderForImageAnalysis(subset, config, shuffle=shuffle, transform=transform)
                dataloaders.append(dataloader)
                # Reset for the next batch
                current_batch_indices = []
                current_pixel_count = 0

        # Add image index to the current batch
        current_batch_indices.append(index)
        current_pixel_count += num_pixels

    # Don't forget to add the last batch if it hasn't been added yet
    if current_batch_indices:
        subset = Subset(image_list, current_batch_indices)
        dataloader = obtainDataloaderForImageAnalysis(subset, config, shuffle=shuffle, transform=transform)
        dataloaders.append(dataloader)

    return dataloaders


def padResize(images, chunk_size, mode):
    padded_images = []
    for image in images:

        pad_x = chunk_size[-2] - image.size()[-2] % chunk_size[-2] if image.size()[-2] % chunk_size[-2] else 0
        pad_y = chunk_size[-1] - image.size()[-1] % chunk_size[-1] if image.size()[-1] % chunk_size[-1] else 0
        if pad_x == 0 and pad_y == 0:
            padded_images.append(image)
            continue         
        
        if mode == 'stretch':
            image = F.interpolate(image.unsqueeze(0), size=(image.size()[1] + pad_x, image.size()[2] + pad_y), mode='bilinear').squeeze(0)
        elif mode == 'crop':
            image = image[:, :image.size()[1] - image.size()[1] % chunk_size[0], :image.size()[2] - image.size()[2] % chunk_size[1]]
        elif mode == 'pad':
            image = F.pad(image, (pad_x//2, pad_x - pad_x//2, pad_y//2, pad_y - pad_y//2), mode='reflect')

        elif mode == 'auto':
            if pad_x <= image.size()[-2] // 2:
                image = F.pad(image, (pad_x//2, pad_x - pad_x//2, 0, 0), mode='reflect')
            else:
                image = image[:, :image.size()[-2] - image.size()[-2] % chunk_size[-2], :]
            if pad_y <= image.size()[-1] // 2:
                image = F.pad(image, (0, 0, pad_y//2, pad_y - pad_y//2), mode='reflect')
            else:
                image = image[:, :, :image.size()[-1] - image.size()[-1] % chunk_size[-1]]

        padded_images.append(image)
    return padded_images

def slice_and_separate_channels(image, chunk_size, image_id):
    C, H, W = image.shape[-3], image.shape[-2], image.shape[-1]
    #chunk_height, chunk_width = chunk_size
    chunk_width, chunk_height = chunk_size #FIX THIS CONSISTENCY PROBLEM PLEASE
    rows = H // chunk_height
    cols = W // chunk_width

    separated_channels = []
    slice_id = 0
    for i in range(rows):
        for j in range(cols):
            start_x = j * chunk_width
            start_y = i * chunk_height

            for c in range(C):
                sub_image = image[0, c:c+1, i*chunk_height:(i+1)*chunk_height, j*chunk_width:(j+1)*chunk_width]
                if sub_image.shape[-1] == 0 or sub_image.shape[-2] == 0:
                    print(f"Warning: Found zero dimension in sub_image at slice_id {slice_id}, channel_id {c}")

                separated_channels.append({
                    'image_id': image_id,
                    'slice_id': slice_id,
                    'channel_id': c,
                    'sub_image': sub_image,
                    'x_location': start_x,
                    'y_location': start_y  # Adding location metadata
                })

            slice_id += 1
    return separated_channels

class CustomInputDataset(Dataset):
    def __init__(self, tensor_images, network_size, mode='auto', transform=None):
        self.network_size = network_size
        self.tensor_images = tensor_images
        self.transform = transform
        self.prepared_data = self.prepare_data(tensor_images, network_size, mode)
        

    def prepare_data(self, tensor_images, network_size, mode):
        prepared_data = []
        image_id = 0
        for tensor_image in tensor_images:
            processed_images = padResize([tensor_image], network_size, mode)
            for processed_image in processed_images:
                prepared_data.extend(slice_and_separate_channels(processed_image, network_size, image_id))
            image_id += 1
        return prepared_data

    def __getitem__(self, index):
        item = self.prepared_data[index]
        if self.transform:
            transformed_image = self.transform(item['sub_image'])
            
            return {**item, 'sub_image': transformed_image}
        return item

    def __len__(self):
        return len(self.prepared_data)
    

def obtain_custom_input_dataset(tensor_images, config, mode='auto', transform = None):
    dataset = CustomInputDataset(tensor_images, (config.width, config.height), mode, transform)   
    return dataset

def probabilities_table(output, charset=default_config.charset):
    probabilities = F.softmax(output, dim=0)
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    return sorted_probs, sorted_indices
def obtainDataloaderForImageAnalysis(tensor_images, config, shuffle=False, transform=None):
    dataset = obtain_custom_input_dataset(tensor_images, config, mode='auto', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    return dataloader
def enrich_and_reorganize_images(dataloader, config):
    
    grouped_data_by_image = defaultdict(lambda: defaultdict(list))
    
    
    for batch in dataloader:
        inputs = batch['sub_image']
        slice_ids = batch['slice_id']
        channel_ids = batch['channel_id']
        image_ids = batch['image_id']
        xlocations = batch['x_location']
        ylocations = batch['y_location']
        outputs = config.model(inputs)
            
        for input_img, output, slice_id, channel_id, x_location, y_location, image_id in zip(inputs, outputs, slice_ids, channel_ids, xlocations, ylocations, image_ids):
                
            sorted_probs, sorted_indices = probabilities_table(output, config.charset)
                
            enriched_item = {
                'image_id': image_id,
                'slice_id': slice_id.item(),
                'channel_id': channel_id.item(),
                'x_location': x_location,
                'y_location': y_location,
                'sorted_probs': sorted_probs,
                'sorted_indices': sorted_indices,
                'sub_image': input_img
            }
                
            grouped_data_by_image[image_id.item()][slice_id.item()].append(enriched_item)
    
    images_data = list(grouped_data_by_image.values())
    return images_data

def load_test_images(size=(1024,1024), items=10, directory=TRAINING_IMAGES_DIRECTORY):
    images = []
    metadata = []
    if not os.path.exists(directory):
        return images, metadata

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                meta = json.load(f)
                if tuple(meta['size']) == size and meta['items'] == items:
                    image_filename = filename.replace('.json', '.png')
                    image_path = os.path.join(directory, image_filename)
                    images.append(image_path)
                    metadata.append(meta)
    return images, metadata

def save_test_image(image, metadata, directory=TRAINING_IMAGES_DIRECTORY):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate a unique filename based on the current timestamp and a random integer
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_int = random.randint(1,100)
    base_filename = f"{timestamp}{random_int}.png"
    json_filename = f"{timestamp}{random_int}.json"
    image_path = os.path.join(directory, base_filename)
    json_path = os.path.join(directory, json_filename)

    pil_image = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype('uint8'), 'L')
    pil_image.save(image_path, 'PNG')

    with open(json_path, 'w') as f:
        json.dump(metadata, f)

    return image_path, json_path

def reset_tensor_values(image, oldvalues, newvalues):
    old_values = set(oldvalues)
    new_values = set(newvalues)

    value_mapping = {}
    used_new_values = set()

    for old_value in old_values:
        if old_value not in new_values:
            # Find a new value that hasn't been used, or use any if all are used
            available_values = list(new_values - used_new_values)
            if available_values:
                new_value = random.choice(available_values)
            else:
                new_value = random.choice(list(new_values))
            used_new_values.add(new_value)
            value_mapping[old_value] = new_value
        else:
            value_mapping[old_value] = old_value  # No change needed

    # Apply the mapping to the image tensor
    for old_value, new_value in value_mapping.items():
        image[image == old_value] = new_value

    return image, list(new_values)
def resize_image(image, size, mode='bicubic', antialias=True):
    return F.interpolate(image, size, mode=mode, align_corners=None if mode == 'bilinear' else None, recompute_scale_factor=False, antialias=antialias)

def generate_test_images(size=(1024,1024), length=25, items=10, values=None, use_cache=False, resizing_factor=8, antialias=True, cache_folder="images", return_metadata=False):
    ultimate_size = size
    ultimate_height, ultimate_width = ultimate_size
    height, width = [ size[0] // resizing_factor, size[1] // resizing_factor ]
    size = [ height, width ]
    mindim = min(height, width)
    margin = max(int(mindim * 0.01),1)  # Margin for shapes to reside within
    
    available_test_images = []
    available_test_images_metadata = []

    if use_cache:
        available_test_images, available_test_images_metadata = load_test_images(ultimate_size, items) 
    
    test_images = torch.zeros((length, 1, height, width), device=device)
    test_images_metadata = []

    # Helper functions to draw shapes

    def draw_circle(image, center_x, center_y, radius, value):
        for x in range(width):
            for y in range(height):
                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                    image[0, y, x] = value

    def draw_square(image, x_min, y_min, side_length, value):
        for x in range(x_min, x_min + side_length):
            for y in range(y_min, y_min + side_length):
                if x == x_min or x == x_min + side_length - 1 or y == y_min or y == y_min + side_length - 1:
                    image[0, y, x] = value
    def draw_filled_square(image, x_min, y_min, side_length, value):
        for x in range(x_min, x_min + side_length):
            for y in range(y_min, y_min + side_length):
                image[0, y, x] = value 
    def draw_triangle(image, point1, point2, point3, value):
        # Use Bresenham's line algorithm for efficiency
        def bresenham(x1, y1, x2, y2):
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            while True:
                image[0, y1, x1] = value  # Set the pixel value

                if x1 == x2 and y1 == y2:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

    def draw_filled_triangle(image, point1, point2, point3, value):
        # Sort points by y-coordinate for efficient filling
        points = sorted([point1, point2, point3], key=lambda p: p[1])
        y_min, y_mid, y_max = points[0][1], points[1][1], points[2][1]

        # Iterate through y-coordinates and fill horizontal lines
        for y in range(y_min, y_max + 1):
            x_start, x_end = None, None

            # Use linear interpolation to find endpoints for each line
            for i in range(3):
                p1, p2 = points[i], points[(i + 1) % 3]
                if p1[1] <= y < p2[1] or p2[1] <= y < p1[1]:
                    if p1[1] != p2[1]:  # Avoid division by zero
                        slope = (p2[0] - p1[0]) / (p2[1] - p1[1])
                        x = p1[0] + slope * (y - p1[1])
                        if x_start is None or x < x_start:
                            x_start = int(x)
                        if x_end is None or x > x_end:
                            x_end = int(x)

            # Sanity check in case of numerical errors
            if x_start is not None and x_end is not None:
                image[0, y, x_start:x_end + 1] = value 
    
    def draw_lined_circle(image, center_x, center_y, radius, value):
        # Use Bresenham's circle algorithm
        x = 0
        y = radius
        d = 3 - 2 * radius
        while y >= x:
            _draw_circle_points(image, center_x, center_y, x, y, value)
            x += 1
            if d > 0:
                y -= 1
                d = d + 4 * (x - y) + 10
            else:
                d = d + 4 * x + 6
    def get_triangle_size(point1, point2, point3):
        side_lengths = [
            ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5,
            ((point3[0] - point2[0])**2 + (point3[1] - point2[1])**2)**0.5,
            ((point1[0] - point3[0])**2 + (point1[1] - point3[1])**2)**0.5,
        ]
        longest_side = max(side_lengths)
        # Using Heron's formula to calculate the area, from which we can derive height
        semi_perimeter = sum(side_lengths) / 2
        area = abs(semi_perimeter * (semi_perimeter - side_lengths[0]) * 
                (semi_perimeter - side_lengths[1]) * (semi_perimeter - side_lengths[2])) ** 0.5
        height = 2 * area / (longest_side+.0001) 
        return longest_side * height
    def _draw_circle_points(image, cx, cy, x, y, value):
        image[0, cy + y, cx + x] = value
        image[0, cy - y, cx + x] = value
        image[0, cy + y, cx - x] = value
        image[0, cy - y, cx - x] = value
        image[0, cy + x, cx + y] = value
        image[0, cy - x, cx + y] = value
        image[0, cy + x, cx - y] = value
        image[0, cy - x, cx - y] = value


    return_array = []

    if not values:
        values = [0.0, 1.0]

    # Generate images
    for i in range(length):
        if len(available_test_images) > i:
            #add condition to check for identical value maps in proposed test image
            available_test_image_array = np.array(Image.open(available_test_images[i]).convert('L'))
            available_test_image = torch.from_numpy(available_test_image_array).float() / 255.0
            available_test_images[i], available_test_images_metadata[i]['values'] = reset_tensor_values(available_test_image, available_test_images_metadata[i]['values'], values)
            return_array.append(available_test_images[i].unsqueeze(0).unsqueeze(0).to(device))
            test_images_metadata.append(available_test_images_metadata[i])
            
            continue
        background_value = random.choice(values)
        remaining_values = [value for value in values if value != background_value]
        image = torch.full((1, height, width), background_value, device=device)
        image_metadata = {
            'values': [ background_value ],
            'size': size,
            'items': items,
            'shapes': []
        }
        
        for _ in range(items):
            image_metadata_shape_subitem = {}
            shape_type = random.choice(implemented_shapes)
            image_metadata_shape_subitem['shape_type'] = shape_type
            value = random.choice(remaining_values)
            image_metadata_shape_subitem['value'] = value

            if value not in image_metadata['values']:
                image_metadata['values'].append(value)
            

            threshold = 10
            image_metadata_shape_subitem['size'] = 0

            if shape_type == "triangle" or shape_type == "filled_triangle":
                while image_metadata_shape_subitem['size'] < threshold:
                    pone_x = random.randint(margin + 1, width - margin - 1)
                    pone_y = random.randint(margin + 1, height - margin - 1)
                    ptwo_x = random.randint(margin + 1, width - margin - 1)
                    ptwo_y = random.randint(margin + 1, height - margin - 1)
                    pthree_x = random.randint(margin + 1, width - margin - 1)
                    pthree_y = random.randint(margin + 1, height - margin - 1)
                    
                    image_metadata_shape_subitem['x_location'] = pone_x
                    image_metadata_shape_subitem['y_location'] = pone_y
                    point_one = (pone_x, pone_y)
                    point_two = (ptwo_x, ptwo_y)
                    point_three = (pthree_x, pthree_y)
                    image_metadata_shape_subitem['size'] = get_triangle_size(point_one, point_two, point_three)

                    if image_metadata_shape_subitem['size'] > threshold:
                        if shape_type == "triangle":
                            draw_triangle(image, point_one, point_two, point_three, value)                            
                        else:
                            draw_filled_triangle(image, point_one, point_two, point_three, value)    
            elif shape_type == "circle" or shape_type == "filled_circle":
                while image_metadata_shape_subitem['size'] < threshold:
                    center_x = random.randint(margin * 2 + 1, width - margin * 2 - 1)
                    center_y = random.randint(margin * 2 + 1, height - margin * 2 - 1)
                    radius = random.randint(margin, min(center_x - margin, width - center_x - margin, center_y - margin, height - center_y - margin))

                    image_metadata_shape_subitem['x_location'] = center_x
                    image_metadata_shape_subitem['y_location'] = center_y
                    image_metadata_shape_subitem['size'] = radius

                    if image_metadata_shape_subitem['size'] > threshold:
                        if shape_type == "circle":
                            draw_lined_circle(image, center_x, center_y, radius, value)            
                        else:
                            draw_circle(image, center_x, center_y, radius, value)
            else:
                while image_metadata_shape_subitem['size'] < threshold:
                    x_min = random.randint(margin + 1, width - margin * 2 - 1)
                    y_min = random.randint(margin + 1, height - margin * 2 - 1)
                    side_length = random.randint(margin, min(width - x_min - margin, height - y_min - margin))
                    image_metadata_shape_subitem['x_location'] = x_min
                    image_metadata_shape_subitem['y_location'] = y_min
                    image_metadata_shape_subitem['size'] = side_length

                    if image_metadata_shape_subitem['size'] > threshold:
                        if shape_type == "square":
                            draw_square(image, x_min, y_min, side_length, value)
                        else:
                            draw_filled_square(image, x_min, y_min, side_length, value)
        
            image_metadata['shapes'].append(image_metadata_shape_subitem)
        
        test_images_metadata.append(image_metadata)
    
        test_images[i] = image.unsqueeze(0).unsqueeze(0)
    
        if use_cache:
            save_test_image(image.unsqueeze(0), test_images_metadata[-1])
    
    resized_images = resize_image(test_images, (ultimate_height, ultimate_width))
    new_return_array = [resized_images[i].unsqueeze(0) for i in range(resized_images.size(0))]

    return_array.extend(new_return_array)
    if return_metadata:
        return return_array, test_images_metadata
    return return_array


def load_images_as_tensors(device, images=None, directory="images", max_size=(2046,2046)):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_tensors = []
    scan_directory = os.getcwd()
    scan_directory = os.path.join(scan_directory, directory)
    if images is None:
        for filename in os.listdir(scan_directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(scan_directory, filename)
                try:
                    image = Image.open(image_path)
                    image_tensor = pilToPreASCIITensor(image, device, max_size=max_size)
                    image_tensors.append(image_tensor.to(device))
                except Exception as e:
                    print(f"Failed to process {image_path}: {str(e)}")
    else:
        for image in images:
            try:
                image_tensor = pilToPreASCIITensor(image, device, max_size=max_size)
                image_tensors.append(image_tensor.to(device))
            except Exception as e:
                print(f"Failed to process provided image: {str(e)}")

    return image_tensors

def pilToPreASCIITensor(image, device=device, max_size=max_size):
    max_width, max_height = max_size
    original_width, original_height = image.size

    # Determine the scaling factor to fit the image within the maximum dimensions while preserving the aspect ratio
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height
    scale_ratio = min(width_ratio, height_ratio)

    # Check if the image dimensions exceed the max allowed dimensions
    if scale_ratio < 1:
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        new_size = (new_width, new_height)
        image = image.resize(new_size, Image.LANCZOS)  # Resize the image with high quality

    # Convert the image to grayscale ('L' for luminance)
    image = image.convert('L')

    # Convert the PIL image to a PyTorch tensor, add a batch dimension, and move to the specified device
    return transforms.ToTensor()(image).unsqueeze(0).to(device)
    

def pilToASCII(image, config, max_size=None):
    if not max_size:
        return tensorToASCII(pilToPreASCIITensor(image), config)
    else:
        return tensorToASCII(pilToPreASCIITensor(image, max_size=max_size), config)
def process_tensor_images(tensor_images, config):
    dataloader = obtainDataloaderForImageAnalysis(tensor_images, config)
    images_data = enrich_and_reorganize_images(dataloader, config)
    return dataloaderToStructuredData(dataloader, config)


def tensorToASCII(img_tensor, config):
    structured_data = process_tensor_images([img_tensor], config)
    ascii_text = ""
    last_y_location = None

    for item in structured_data[0]: 
        y_location = item['y_location'].item()
        if last_y_location is None or last_y_location != y_location:
            ascii_text += '\n' if last_y_location is not None else ""
            last_y_location = y_location

        char_id = item['sorted_indices'][0].item()
        char = config.charset[char_id]
        ascii_text += char

    return ascii_text
import textwrap
def layout_strings(blocks, width):
    # Calculate dimensions of each block
    block_dimensions = []
    for block in blocks:
        lines = block.split('\n')
        height = len(lines)
        max_width = max(len(line) for line in lines)
        block_dimensions.append((height, max_width, block))
    
    # Sort blocks by height (tallest first)
    block_dimensions.sort(reverse=True, key=lambda x: x[0])
    
    # Prepare to layout the blocks
    result = []
    current_line = []
    current_height = 0
    line_width = 0
    
    # Place blocks
    for height, max_width, block in block_dimensions:
        # Check if block fits on the current line
        if line_width + max_width <= width:
            current_line.append(block)
            line_width += max_width
            current_height = max(current_height, height)
        else:
            # Flush current line
            if current_line:
                result.append(current_line)
            # Start new line with the current block
            current_line = [block]
            line_width = max_width
            current_height = height
    
    # Append the last line if not empty
    if current_line:
        result.append(current_line)
    
    # Format the result for printing
    formatted_result = format_blocks(result, width)
    return formatted_result

def format_blocks(lines, width):
    # Interleave lines from different blocks and fill space
    formatted_lines = []
    for line_blocks in lines:
        # Determine the height of the tallest block in this row
        max_height = max(block.count('\n') + 1 for block in line_blocks)
        # Create buffers for each line in this row
        buffers = ['' for _ in range(max_height)]
        for block in line_blocks:
            block_lines = block.split('\n')
            block_height = len(block_lines)
            for i in range(block_height):
                buffers[i] += block_lines[i].ljust(width // len(line_blocks))
            for i in range(block_height, max_height):
                buffers[i] += ' ' * (width // len(line_blocks))
        # Join buffers to form the full lines
        formatted_lines.extend(buffers)
    return '\n'.join(formatted_lines)


def dataloaderImageToImage(dataloader, config, live=False, size=None):
    structured_data = dataloaderToStructuredData(dataloader, config)
    ascii_representations = structuredDataToASCII(structured_data, config)
    grouped_ascii_representations = layout_strings(ascii_representations, 250)
    
    if live:
        print(grouped_ascii_representations)

    return renderArrayOfASCII(ascii_representations, config, size)

def dataloaderToStructuredData(dataloader, config):
    images_data = enrich_and_reorganize_images(dataloader, config)
    structured_data = []
    for image_data in images_data:
        all_items = []
        for slice_list in image_data.values():
            all_items.extend(slice_list)  # Flatten all slices into a single list

        sorted_items = sorted(all_items, key=lambda x: (x['y_location'].item(), x['x_location'].item()))
        structured_data.append(sorted_items)

    return structured_data    

def structuredDataToASCII(structured_data, config):
    ascii_representations = []
    for data in structured_data:
        ascii_representation = ""
        last_y_location = None
        for item in data:
            y_location = item['y_location'].item()
            if last_y_location is None or last_y_location != y_location:
                if last_y_location is not None:
                    ascii_representation += '\n'
                last_y_location = y_location

            char_id = item['sorted_indices'][0].item()
            char = config.charset[char_id]
            ascii_representation += char
        ascii_representations.append(ascii_representation)
    return ascii_representations
def renderArrayOfASCII(ascii_array, config, size=None):
    output_image_tensors = []
    for ascii_representation in ascii_array:

        ascii_image_tensor = render_text_lines(config.charBitmasks, ascii_representation, config.charset, config.width, config.height)
        image_tensor = text_tensor_to_image_tensor(ascii_image_tensor)
        if size != None:
            resized_output_image_tensor = F.interpolate(image_tensor.unsqueeze(0).unsqueeze(0).float(), size, mode='bilinear', align_corners=False)
            output_image_tensors.append(resized_output_image_tensor)
        else:
            output_image_tensors.append(image_tensor)

    return output_image_tensors

def tensorsToASCII(tensor_images, config, live=False):
    structured_data = process_tensor_images(tensor_images, config)
    ascii_representations = structuredDataToASCII(structured_data, config)
    return renderArrayOfASCII(ascii_representations, config)

def compute_image_loss(image_tensor, processed_tensor, mode='average'):
    """
    Computes a combined loss score based on selected mode.

    Args:
        image_tensor (Tensor): The original image tensor.
        processed_tensor (Tensor): The processed image tensor to compare.
        mode (str): The mode of combining losses, options are 'average', 'mse', 'ssim', 'cosine'.

    Returns:
        float: Normalized loss score between 0 and 1.
    """
    # Convert tensors to CPU and numpy, ensure they are float type
    image_np = image_tensor.squeeze().to('cpu').numpy().astype(np.float32)
    processed_np = processed_tensor.squeeze().to('cpu').numpy().astype(np.float32)

    # MSE Loss
    mse_loss = F.mse_loss(image_tensor, processed_tensor)

    ssim_score = 0
    if COLOR_MIXING_AVAILABLE and ssim is not None:
        if image_np.ndim == 3:  # For RGB or similar
            image_np = image_np.mean(axis=0)  # Convert to grayscale
        if processed_np.ndim == 3:
            processed_np = processed_np.mean(axis=0)

        ssim_loss = ssim(
            image_np,
            processed_np,
            data_range=processed_np.max() - processed_np.min(),
        )
        ssim_score = (ssim_loss + 1) / 2  # Normalize SSIM to range from 0 to 1
    
    
    # Cosine Similarity
    cos_sim = F.cosine_similarity(image_tensor.flatten().unsqueeze(0), processed_tensor.flatten().unsqueeze(0), dim=1)
    cos_sim = (cos_sim + 1) / 2  # Normalize cosine similarity to range from 0 to 1
    

    # Compute combined loss based on mode
    if mode == 'average':
        return (mse_loss + ssim_score + cos_sim) / 3
    elif mode == 'mse':
        return mse_loss
    elif mode == 'ssim':
        return ssim_score
    elif mode == 'cosine':
        return cos_sim
    else:
        raise ValueError("Invalid mode selected. Choose 'average', 'mse', 'ssim', or 'cosine'.")

# Ensure the tensors are correctly shaped and converted for the function calls.
def process_tensor(config, tensor, size=None):
    dataloader = create_dataloaders([tensor], config, image_batch_limit, shuffle=False)[0]
    processed_tensor = dataloaderImageToImage(dataloader, config, False)[0]
    if size:
        processed_tensor = resize_image(processed_tensor.unsqueeze(0).unsqueeze(0).float()/255, size)
    return processed_tensor

if PYQT_AVAILABLE:
    class FeedbackWidget(QWidget):
        def __init__(self, config, image_tensor, metadata):
            super().__init__()
            self.config = config
            self.image_tensor = image_tensor
            self.metadata = metadata
            self.setWindowTitle(metadata.get('title', 'Feedback'))
            self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Set up the image display and processing
        self.processed_tensor = process_tensor(self.config, self.metadata['correct_tensor'].clone().squeeze(0), (512, 512))
        
        # Setup layout components
        self.setup_header(layout)
        self.setup_images(layout, self.image_tensor, self.processed_tensor)
        self.setup_footer(layout)
        self.setup_controls(layout)
        
        self.setLayout(layout)

        # Setup auto-submit feature
        self.setup_auto_submit()



    def setup_header(self, layout):
        header_label = QLabel(self.metadata.get('header', ''))
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

    def setup_images(self, layout, image_tensor, processed_tensor):
        image_layout = QHBoxLayout()

        self.reference_image_label = QLabel(self)
        self.reference_image_label.setPixmap(self.tensor_to_qpixmap(processed_tensor.squeeze(0).squeeze(0)))
        image_layout.addWidget(self.reference_image_label)

        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.tensor_to_qpixmap(image_tensor.squeeze(0).squeeze(0)))
        image_layout.addWidget(self.image_label)

        layout.addLayout(image_layout)

    def setup_footer(self, layout):
        footer_label = QLabel(self.metadata.get('footer', ''))
        footer_label.setWordWrap(True)
        layout.addWidget(footer_label)

    def setup_controls(self, layout):
        control_layout = QHBoxLayout()
        inverted_mse_loss = 1 - compute_image_loss(self.image_tensor, self.processed_tensor)
        self.percentage_label = QLabel(f"{inverted_mse_loss.item() * 100:.1f}%")
        self.percentage_label.setFont(QFont('Arial', 14, QFont.Bold))
        control_layout.addWidget(self.percentage_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(10000)
        self.slider.setValue(int(self.slider.maximum() * inverted_mse_loss.item()))
        self.slider.valueChanged.connect(self.update_percentage)
        control_layout.addWidget(self.slider)

        self.submit_button = QPushButton('Submit')
        self.submit_button.clicked.connect(self.submit)
        control_layout.addWidget(self.submit_button)

        layout.addLayout(control_layout)

    def setup_auto_submit(self):
        self.timer = QTimer(self)
        self.timer.setInterval(5000)  # 5000 ms = 5 seconds
        self.timer.timeout.connect(self.submit)
        self.timer.start()
        self.slider.valueChanged.connect(self.timer.stop)  # Stop timer if slider value changes

    def update_percentage(self, value):
        percentage = value / self.slider.maximum() * 100
        self.percentage_label.setText(f"{percentage:.1f}%")

    def tensor_to_qpixmap(self, tensor):
        array = tensor.to("cpu").numpy() * 255
        array = array.astype(np.uint8)
        image = Image.fromarray(array, 'L')
        qimage = QImage(image.tobytes(), image.width, image.height, image.width, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)

    def submit(self):
        print("Submitting data...")
        self.slider_value = self.slider.value()
        self.close()

    def show_image_with_feedback(config, image_tensor, metadata):
        app = QApplication(sys.argv)
        feedback_widget = FeedbackWidget(config, image_tensor.squeeze(0), metadata)
        if human_mode == 'semi-auto' or human_mode == 'manual':
            feedback_widget.show()
            app.exec_()
        else:
            return torch.tensor(float(feedback_widget.slider.value()), requires_grad=True, device=device)

        return torch.tensor(float(feedback_widget.slider_value), requires_grad=True, device=device)
else:
    def show_image_with_feedback(*args, **kwargs):
        raise RuntimeError("PyQt5 is required for human feedback UI")
def collect_human_input(config, data, metadata, category, method = 'logits', slider_max=10000):
    """
    Collects human feedback and optionally returns a probability distribution.

    Args:
        data: A list of image tensors.
        metadata: Any additional category-specific metadata.
        category: The name of the category.
        normalize: If True (default), normalize scores into a probability distribution.
                   If False, return scores normalized by the slider's maximum value.
        slider_max: The maximum value of the slider (default: 10000).
    """
    print(f"Category: {category}")
    human_responses = []
    display_metadata = {"title":category,"header":f"This image is of type {category}","footer":f"Set the slider to the right if the image resembles the reference image, to the left if it definitely isn't the reference image."}
    display_metadata['possible_labels'] = metadata['possible_labels']
    display_metadata['correct_label'] = metadata['correct_label']
    display_metadata['correct_tensor'] = data[display_metadata['correct_label']]

    for image_tensor in data:
        response = show_image_with_feedback(config, image_tensor, display_metadata)
        human_responses.append(response)
        
    if method == 'probabilities':
        # Normalize into probability distribution (with edge case handling)
        responses_array = np.array(human_responses) 
        sum_of_responses = responses_array.sum()

        if sum_of_responses == 0:
            # Handle zero-sum case
            probabilities = np.ones_like(responses_array) / len(responses_array)  # Equal probabilities
        else:
            probabilities = responses_array / sum_of_responses
        return probabilities
    if method == 'normalized' or method == 'logits':
        # Normalize by slider maximum value
        return_value = [response / slider_max for response in human_responses]
    if method == 'logits':
        # -10 to 10 scale for typical logit range
        return_value = torch.tensor(return_value, requires_grad=True, device=device) * 20 - 10
    else:
        return response
    return return_value
from PIL import ImageSequence

def split_and_process_rgb(image, display_image_func, config):
    """ Split the image into RGB components, process each with display_image_func, and blend. Handle GIF by processing each frame. """
    
    # Check if the image is animated (contains multiple frames)
    if hasattr(image, "is_animated") and image.is_animated:
        frames = []
        for frame in ImageSequence.Iterator(image):
            frame = frame.copy()
            processed_frame = process_frame(frame, display_image_func, config)
            frames.append(processed_frame.convert('P', palette=Image.ADAPTIVE))
        # Reassemble the processed frames into a new GIF
        out = io.BytesIO()
        processed_image = frames[0]
        processed_image.info = image.info  # Preserve GIF metadata like loop
        processed_image.save(out, format='GIF', save_all=True, append_images=frames[1:], loop=0)
        processed_image.save("out,gif", format='GIF', save_all=True, append_images=frames[1:], loop=0)
        out.seek(0)
        return Image.open(out)
    else:
        # Handle non-animated images (single frame processing)
        return process_frame(image, display_image_func, config)

def process_frame(frame, display_image_func, config):
    """Process a single frame for RGB and optional alpha handling."""
    num_channels = len(frame.getbands())
    
    if num_channels == 1:
        channels = [frame] * 3
    elif num_channels == 3:
        channels = frame.split()
    elif num_channels == 4:
        r, g, b, a = frame.split()
        channels = [r, g, b]
        alpha = a  # Preserve alpha channel for later recombination

    processed_channels = [display_image_func(config, [ch], True) for ch in channels]

    if num_channels == 4:
        alpha_resized = alpha.resize(processed_channels[0].size, Image.LANCZOS)
        processed_channels.append(alpha_resized)
        return Image.merge('RGBA', tuple(processed_channels))
    else:
        return Image.merge('RGB', tuple(processed_channels))
# Example usage



default_config.model_loaded = False
if load_model(default_config):
    default_config.model_loaded = True

if user_pil:
    split_and_process_rgb(user_pil, display_image, default_config).show()
    #display_image(default_config, [user_pil], True).show()

    ascii_text = pilToASCII(user_pil, default_config)
    print(ascii_text)
    exit()
elif server:
    
    from http.server import BaseHTTPRequestHandler
    import socketserver
    import json
    from PIL import Image
    import io
    import base64
    import cgi

    
    PORT = 8000
    def process_image(img):
        # Load your model and any other initial setup
        load_model(default_config)
        image_format = 'PNG'
        # Process the image (assuming split_and_process_rgb has been adapted for animation)
        processed_img = split_and_process_rgb(img, display_image, default_config)

        # Prepare image for return (store it in a memory buffer)
        img_buffer = io.BytesIO()

        # Determine the best format based on animation and alpha presence
        if hasattr(processed_img, "is_animated") and processed_img.is_animated:
            # For animated images, check if alpha is used
            if processed_img.mode == 'RGBA':
                # Use APNG if alpha transparency is required in animations
                # Note: Ensure your environment supports APNG
                processed_img.save(img_buffer, format='APNG')
                image_format = 'APNG'
            else:
                # Use GIF for animations without alpha or where binary transparency is sufficient
                processed_img.save(img_buffer, format='GIF')
                image_format = 'GIF'
        else:
            # For still images, use PNG to preserve alpha transparency if present
            processed_img.save(img_buffer, format='PNG')

        del processed_img
        img_buffer.seek(0)
        data = {
            'image_data': base64.b64encode(img_buffer.getvalue()).decode("utf-8"),  # Encode image bytes as base64
            'image_format': image_format
        }

        return json.dumps(data).encode('utf-8')  # Encode response as bytes

    


    class ImageProcessHandler(BaseHTTPRequestHandler):
        def do_POST(self):

            if self.path == '/process_image':
                # Assume the POST request contains "multipart/form-data"
                ctype, pdict = cgi.parse_header(self.headers['content-type'])
                if ctype == 'multipart/form-data':
                    pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                    pdict['CONTENT-LENGTH'] = int(self.headers['Content-Length'])
                    fields = cgi.parse_multipart(self.rfile, pdict)
                    if 'image_file' in fields:
                        # Extract the image file data
                        file_data = fields['image_file'][0]
                        try:
                            img = Image.open(io.BytesIO(file_data))
                            processed_data = process_image(img)
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.send_header('Access-Control-Allow-Origin', '*')  # Allow all domains
                            self.end_headers()

                            self.wfile.write(processed_data)
                        except Exception as e:
                            self.send_error(500, f"Error processing image: {str(e)}")
                    else:
                        self.send_error(400, "Image file not found in the request.")
                else:
                    self.send_error(400, "Wrong content type.")
        def do_OPTIONS(self):
            # Handle pre-flight requests for CORS
            self.send_response(200, "ok")
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

    with socketserver.TCPServer(("", PORT), ImageProcessHandler) as httpd:
        print("Serving at port", PORT)
        httpd.serve_forever()



else:

    default_config.metamodel = MetaModel()
    default_config.metamodel = load_model(None, meta_model=default_config.metamodel)
    train_model(default_config, live)
if NVML_AVAILABLE:
    pynvml.nvmlShutdown()

