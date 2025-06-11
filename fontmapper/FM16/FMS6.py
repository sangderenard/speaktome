#!/usr/bin/env python3
"""FontMapper FM16 utility functions."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

import json
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import platform
import math

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

# --- Add these flags at the top ---
RUN_SERVER = False
TEXT_ONLY = False
INPUT_IMAGE_PATH = None

# --- Parse command line args for text output mode and input image ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FontMapper ASCII/Model Utility")
    parser.add_argument("--text", action="store_true", help="Output ASCII text to stdout instead of running server")
    parser.add_argument("--input", type=str, help="Input image file path for text output mode")
    parser.add_argument("--no-color", action="store_true", help="Disable advanced color channel mixing")
    parser.add_argument("--port", type=int, default=8000, help="Port for server (if running)")
    args = parser.parse_args()
    if args.text:
        TEXT_ONLY = True
        INPUT_IMAGE_PATH = args.input
    else:
        RUN_SERVER = True
    if args.no_color:
        ENABLE_COLOR_MIXING = False
    else:
        ENABLE_COLOR_MIXING = True
    PORT = args.port
else:
    ENABLE_COLOR_MIXING = False  # Default for import

config_file = 'server.yaml'
if len(sys.argv) > 1:
    config_file= sys.argv[-1]

server = user_pil = None
PORT = 8000
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
        globals()[name] = torch.tensor(float(value), requires_grad=True, dtype=torch.float16)
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


FMversion = "FM37"


    

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
            max_height = max(max_height, height + abs(offset_y))
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
        ToTensorAndToDevice(device=device)
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


def resize_image(image, size, mode='bicubic', antialias=True):
    return F.interpolate(image, size, mode=mode, align_corners=None if mode == 'bilinear' else None, recompute_scale_factor=False, antialias=antialias)


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
    index_array = []
    for item in structured_data[0]: 
        y_location = item['y_location'].item()
        if last_y_location is None or last_y_location != y_location:
            ascii_text += '\n' if last_y_location is not None else ""
            last_y_location = y_location

        char_id = item['sorted_indices'][0].item()
        index_array.append(char_id)
        char = config.charset[char_id]
        ascii_text += char

    return ascii_text, index_array
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

def blend_tensors(tensor1, tensor2, operation):
    if operation == 'add':
        return torch.clamp(tensor1 + tensor2, 0, 1)
    elif operation == 'multiply':
        return tensor1 * tensor2
    elif operation == 'difference':
        return torch.abs(tensor1 - tensor2)



#predefined_hues = {
#        'red': 0,
#        'orange': 30,
#        'yellow': 60,
#        'green': 120,
#        'cyan': 180,
#        'blue': 240,
#        'violet': 275,
#        'magenta': 300
#    }
predefined_hues = {
    'red': 0,
    'orange': 30,
    'yellow': 60,
    'lime': 90,
    'green': 120,
    'mint': 150,
    'cyan': 180,
    'azure': 210,
    'blue': 240,
    'violet': 275,
    'magenta': 300,
    'rose': 330,
    'red-orange': 15,
    'yellow-orange': 45,
    'yellow-green': 75,
    'green-cyan': 165,
    'cyan-blue': 195,
    'blue-violet': 255,
    'purple': 290, 
    'pink': 350,
    'chartreuse': 90,
    'spring green': 150, 
    'indigo': 270, 
    'amber': 45,
    'teal': 180,
    'vermilion': 10,
    'aquamarine': 160
}

def hue_centers(*args):
    return {arg: predefined_hues.get(arg, arg) for arg in args}


def epsilon(sigma):
    return 10**-3 * sigma**2

def theta(hue_degrees):
    return np.deg2rad(hue_degrees)

CLOCKWISE, COUNTERCLOCKWISE = 0, 1

def min_circular_distance(theta, theta_i):
    # Ensure theta and theta_i are numpy arrays to handle both single values and arrays uniformly
    theta = np.atleast_1d(theta)
    theta_i = np.atleast_1d(theta_i)
    
    # Calculate the sine of the difference
    sines_of_difference = np.sin(theta_i - theta)
    
    # Determine direction based on the sign of the sine
    directions = np.where(sines_of_difference > 0, COUNTERCLOCKWISE, CLOCKWISE)
    
    # Calculate the minimum distance
    absolute_difference = np.abs(theta_i - theta)
    min_distance = np.minimum(absolute_difference, 2 * np.pi - absolute_difference)
    
    # Combine distances and directions into a single array of tuples
    result = np.array(list(zip(min_distance, directions)))
    
    # If the input was a single value, return a single tuple instead of an array of one tuple
    if result.shape[0] == 1:
        return result[0]
    return result


def compute_sigma(sorted_hues, epsilon=.25):
    radians = [theta(deg) for _, deg in sorted_hues]
    print(radians)
    radians.append(radians[0] + 2 * np.pi)  # Wrap around to the first hue
    sigma_clockwise = {}
    sigma_counterclockwise = {}

    for i in range(len(sorted_hues)):
        hue_name = sorted_hues[i][0]
        left_index = (len(sorted_hues) + i - 1) % len(sorted_hues)
        right_index = (i + 1) % len(sorted_hues)

        distance_one, direction_one = min_circular_distance(radians[i], radians[left_index])
        distance_two, direction_two = min_circular_distance(radians[right_index], radians[i])

        # Assign sigma values based on the direction
        if direction_one == CLOCKWISE:
            sigma_clockwise[hue_name] = distance_one / np.sqrt(-2 * np.log(epsilon))
            sigma_counterclockwise[hue_name] = distance_two / np.sqrt(-2 * np.log(epsilon))
        else:
            sigma_clockwise[hue_name] = distance_two / np.sqrt(-2 * np.log(epsilon))
            sigma_counterclockwise[hue_name] = distance_one / np.sqrt(-2 * np.log(epsilon))

    
    return sigma_clockwise, sigma_counterclockwise

def complex_color_merge(hues, channels):
    width, height = channels[0].size
    composite_hsv_img = Image.new('HSV', (width, height))
    composite_pixels = composite_hsv_img.load()

    x_components = np.zeros((height, width))
    y_components = np.zeros((height, width))
    total_influence = np.zeros((height, width))
    sorted_hues = sorted(hues.items(), key=lambda x: x[1])
    sigma_clockwise, sigma_counterclockwise = compute_sigma(sorted_hues)

    output_arrays = {color: np.array(image, dtype=float)/255.0 for color, image in zip(hues.keys(),channels)}
    bandwidth = {color: sigma_clockwise[color] + sigma_counterclockwise[color] for color in hues.keys()}

    for color, degrees, in hues.items():
        radians = np.deg2rad(degrees)
        influence = output_arrays[color]

        x_comp = influence * np.cos(radians) * bandwidth[color]
        y_comp = influence * np.sin(radians) * bandwidth[color]

        x_components += x_comp
        y_components += y_comp
        total_influence += np.sqrt(abs(x_comp**2 + y_comp**2))

    total_influence[total_influence <=0] = .0001
    
    resultant_hue = np.arctan2(y_components, x_components)
    resultant_hue_degrees = np.rad2deg(resultant_hue) % 360
    final_h = (resultant_hue_degrees / 360 * 255).astype(int)

    s_gamma = 2.2
    v_gamma = 2.2

    primary_hue_magnitude = np.sqrt(x_components**2 + y_components**2)
    saturation_magnitude = ((primary_hue_magnitude/ (total_influence))**(1/s_gamma)) * 255
    saturation_magnitude = np.clip(saturation_magnitude, 0, 255)

    channel_arrays = [((np.array(channel, dtype=float)/255.0)) for channel in channels]
    
    v = ((np.max(channel_arrays, axis=0)**(1/v_gamma)+np.mean(channel_arrays, axis=0)**(1/v_gamma))/2) * 255.0

    for y in range(height):
        for x in range(width):
            composite_pixels[x,y] = (final_h[y, x], int(saturation_magnitude[y,x]), int(v[y, x]))

    rgb_data = composite_hsv_img.convert('RGB')

    return rgb_data

def splitbyhues(img, hues):
    """Split an image by hues and apply a Gaussian based on hue-specific sigma values."""
    print(f"split an image by hues: {hues}")
    custom_hue_radians = {color: theta(degrees) for color, degrees in hues.items()}
    sorted_hues = sorted(hues.items(), key=lambda x: x[1])
    sigma_clockwise, sigma_counterclockwise = compute_sigma(sorted_hues)

    # Convert to HSV and process
    img_hsv = img.convert('HSV')
    hsv_data = np.array(img_hsv, dtype=float)
    h, s, v = hsv_data[..., 0] * 360.0 / 255.0, hsv_data[..., 1], hsv_data[..., 2]
    theta_h = theta(h)
    contributions = {degrees: np.zeros(img.size[::-1]) for color, degrees in hues.items()}
    white_influence = 1 - s/255.0
    print(f"white influence: {white_influence}")
    print(f"v: {v}")
    for color, degrees in hues.items():
        theta_color = theta(degrees)  # Convert once
        circular_distances = min_circular_distance(theta_h, theta_color)  # Assuming a vectorized version exists
        distances, directions = zip(*circular_distances)
        min_dists = np.array(distances)
        directions = np.array(directions)
        sigma_values = np.where(directions == CLOCKWISE, sigma_clockwise[color], sigma_counterclockwise[color])
        greys = white_influence * v
        print(greys)
        colors = np.exp(- (min_dists**2) / (2 * sigma_values**2 + epsilon(sigma_values))) * s * v / 255
        print(colors)
        contributions[degrees] =  colors + greys
    output_images = {}
    for degrees in contributions:
        image_data = contributions[degrees].astype(np.uint8)
        output_images[degrees] = Image.fromarray(image_data, 'L')

    return output_images


class Oversampling_Dataset(Dataset):
    def __init__(self, config, image_list, settings):
        self.base_rotation = round(float(settings.get("rotation", 0))) if settings else 0
        self.v_offset = int(settings.get("v_offset", 0)) if settings else 0
        self.h_offset = int(settings.get("h_offset", 0)) if settings else 0
        self.rotation_oversampling = int(settings.get("r_oversampling", 0)) if settings else 0
        self.vertical_oversampling = int(settings.get("v_oversampling", 0)) if settings else 0
        self.horizontal_oversampling = int(settings.get("h_oversampling", 0)) if settings else 0
        self.config = config
        self.image_list = image_list
        
        step = 360 / (self.rotation_oversampling + 1)
        self.rotations = [round(self.base_rotation + step * i) for i in range(0, self.rotation_oversampling+1)]      

        self.v_shift_increment = self.config.height / (self.vertical_oversampling+1)
        self.v_shifts = [self.v_offset + self.v_shift_increment * i for i in range(0, self.vertical_oversampling + 1)]
        
        self.h_shift_increment = self.config.width / (self.horizontal_oversampling+1)
        self.h_shifts = [self.h_offset + self.h_shift_increment * i for i in range(0, self.horizontal_oversampling + 1)]

        self.transformed_images = self._generate_transformed_images()

    def _generate_transformed_images(self):
        transformed_images = []
        for image_idx, image in enumerate(self.image_list):
            image, padding = self._pad(image)
            # Original image
            transformed_images.append((image_idx, 0, 0, 0, np.array(image, dtype=np.uint8), padding, True))  # (original_idx, rot, vert, horiz, image, padding)

            # Calculate list of rotations to be applied based on rotation_oversampling


            for rotation in self.rotations:
                if rotation == 0 and self.v_shifts[0] == 0 and self.h_shifts[0] == 0:
                    transformed_images.append((image_idx, 0, 0, 0, np.array(image.copy(), dtype=np.uint8), padding, False))
                else:
                    rotated_image = self._rotate(image, rotation)
                    transformed_images.append((image_idx, rotation, self.v_shifts[0], self.h_shifts[0], np.array(rotated_image, dtype=np.uint8), padding, False))
            
            # First, apply all vertical shifts
            if len(self.v_shifts) > 1:
                for v_shift in self.v_shifts:
                    shifted_image = self._shift(image, 0, v_shift)

                    for rotation in self.rotations:
                        rotated_image = self._rotate(shifted_image, rotation)
                        transformed_images.append((image_idx, rotation, v_shift, 0, np.array(rotated_image, dtype=np.uint8), padding, False))

            # Next, apply all horizontal shifts
            if len(self.h_shifts) > 1:
                for h_shift in self.h_shifts:
                    shifted_image = self._shift(image, h_shift, 0)

                    # Apply rotations to the shifted image
                    for rotation in self.rotations:
                        rotated_image = self._rotate(shifted_image, rotation)
                        transformed_images.append((image_idx, rotation, 0, h_shift, np.array(rotated_image, dtype=np.uint8), padding, False))


        return transformed_images

    def _pad(self, image):
        # height and width are made perfect multiples of config.height * config.width
        # This permits the images to be correctly padded regardless of rotation

        required_multiple = math.lcm(self.config.height, self.config.width)
        h_pad, v_pad = [ self.config.width - image.size[0] % required_multiple, self.config.height - image.size[1] % required_multiple ]
        h_pad_odd = h_pad % 2
        v_pad_odd = v_pad % 2
        h_pad = h_pad // 2
        v_pad = v_pad // 2
        #left_pad, top_pad, right_pad, bottom_pad
        padding = (h_pad, v_pad, h_pad + h_pad_odd, v_pad + v_pad_odd)
        image = ImageOps.expand(image, padding)

        return image, padding

    def _rotate(self, image, rotation):
        image = image.rotate(rotation, Image.BICUBIC, expand = True)
        return image
    
    def _shift(self, image, h_shift, v_shift):
        h_shift = round(h_shift)
        v_shift = round(v_shift)
        image_size = image.size
        image = image.crop((h_shift, v_shift, image_size[0]+h_shift, image_size[1]+v_shift))
        

        return image
    
    def __len__(self):
        return len(self.transformed_images)

    def __getitem__(self, idx):
        return self.transformed_images[idx]
    
def display_image(config, image_list, out_image=False, channel_settings=None):
    with torch.no_grad():
        print("In display_image")
        if "bypass" in channel_settings and channel_settings["bypass"]:
            return image_list
        if channel_settings.get("invert", False):
            for i, image in enumerate(image_list):
                image_list[i] = ImageOps.invert(image)
        oversampled_dataset = Oversampling_Dataset(config, image_list, channel_settings)
        dataloader = torch.utils.data.DataLoader(oversampled_dataset, batch_size=1) #batch oversampled images to improve performance
        return_images = []
        original_images_data = {}  # Store data for original images
        for batch in dataloader:

            

            original_idx, rotation, v_shift, h_shift, image, padding, original = batch

            
            # Convert to appropriate types if needed
            original_idx = original_idx.tolist()
            rotation = rotation.tolist()
            v_shift = v_shift.tolist()
            h_shift = h_shift.tolist()

            for i in range(len(original_idx)): #batch images to reduce calls
                one_image = Image.fromarray(image[i].numpy(), 'L')
                queue_image_ascii, index_array = pilToASCII(one_image, config, max_size=one_image.size)  # Pass individual image
                print(queue_image_ascii)
                queue_image_image = None
                if out_image:
                    queue_image_text_rendering = text_tensor_to_image_tensor(render_text_lines(config.charBitmasks, queue_image_ascii, config.charset, config.width, config.height))
                    queue_image_image = Image.fromarray(queue_image_text_rendering.numpy(), 'L')
                if original:  
                    original_images_data[original_idx[i]] = {
                        "ascii": queue_image_ascii,
                        "index_array": index_array,
                        "image": queue_image_image,
                        "rotation": 0,
                        "v_shift": 0,
                        "h_shift": 0,
                        "padding": padding[i],
                        "oversampled": []
                    }
                else:
                    # Store oversampled images, associated with their original idx
                    if original_idx[i] not in original_images_data:
                        original_images_data[original_idx[i]] = {"oversampled": []}
                    
                    original_images_data[original_idx[i]]["oversampled"].append({
                        "ascii": queue_image_ascii,
                        "index_array": index_array,
                        "image": queue_image_image,
                        "rotation": rotation[i],
                        "v_shift": v_shift[i],
                        "h_shift": h_shift[i],
                        "padding": padding[i]
                    })


        # Process the oversampled images
        images = process_oversampled_images(original_images_data)

        # Iterate over the values in the returned dictionary
        for image_data in images.values():
            return_images.append(image_data["blended_image"])
        
        return return_images


import gzip
import pickle
import io
import sys

def process_oversampled_images(original_images_data):
    """Processes and blends oversampled images back into the original image."""
    for original_idx, image_data in original_images_data.items():
        original_image = image_data["image"]
        original_ascii = image_data["ascii"]
        num_oversampled_images = len(image_data["oversampled"])
        if(num_oversampled_images == 0):
            exit()
        # Create a blank image with an alpha channel for blending
        blended_image = Image.new("RGBA", (original_image.size[0], original_image.size[1]), (0, 0, 0, 0))

        # Prepare data structures separately:
        index_array_data = {
            "index_array": image_data["index_array"],
            "rotation": 0,
            "v_shift": 0,
            "h_shift": 0,
            "padding": image_data["padding"]
        }
        ascii_array_data = {
            "ascii": original_ascii,
            "rotation": 0,
            "v_shift": 0,
            "h_shift": 0,
            "padding": image_data["padding"]
        }
        
        # Process each oversampled image, including the original
        for entry in image_data["oversampled"]:
            oversampled_image = entry["image"]
            rotation = entry["rotation"]
            v_shift = entry["v_shift"]
            h_shift = entry["h_shift"]

            # Derotate and deshift oversampled image
            oversampled_image = oversampled_image.rotate(-rotation, expand=True)
            if rotation % 90 != 0:
                old_size = original_image.size
                new_size = oversampled_image.size

                left = (new_size[0] - old_size[0]) // 2
                top = (new_size[1] - old_size[1]) // 2
                right = left + old_size[0]
                bottom = top + old_size[1]

                oversampled_image = oversampled_image.crop((left, top, right, bottom))

            oversampled_image = oversampled_image.crop((
                -h_shift, -v_shift, oversampled_image.size[0] - h_shift, oversampled_image.size[1] - v_shift))

            # Calculate alpha for blending (transparency)
            alpha = 1 / num_oversampled_images

            # Convert alpha channel to a mask image
            oversampled_image = oversampled_image.convert("RGBA").resize(original_image.size, Image.BICUBIC)

            # Blend the oversampled image onto the blended image using the alpha mask
            blended_image = ImageChops.add(blended_image, oversampled_image, scale=alpha, offset=0)

        # Update original image data with blended image
        original_images_data[original_idx] = {
            "ascii": original_ascii,
            "blended_image": blended_image,
            "padding": image_data["padding"]
        }

        # Serialize and gzip-compress the data structures
        serialized_index_array_data = pickle.dumps(index_array_data)
        compressed_index_array_data = gzip.compress(serialized_index_array_data)

        serialized_ascii_array_data = pickle.dumps(ascii_array_data)
        compressed_ascii_array_data = gzip.compress(serialized_ascii_array_data)

        # Calculate the size of the original image in memory
        image_buffer = io.BytesIO()
        original_image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        original_image_data = image_buffer.getvalue()

        # Compress the original image data with gzip
        compressed_image_data = gzip.compress(original_image_data)

        # Calculate the size of original_ascii
        original_ascii_size = sys.getsizeof(original_ascii)
    
        # Convert ASCII string to bytes
        ascii_bytes = original_ascii.encode('utf-8')

        # Compress the ASCII data
        compressed_ascii_data = gzip.compress(ascii_bytes)

        # Calculate the compressed ASCII size
        gzipped_ascii_size = len(compressed_ascii_data)

        # Embed stats into the original_images_data dictionary
        original_images_data[original_idx]["stats"] = {
            "index_array_size": sys.getsizeof(index_array_data),
            "gzipped_index_array_size": len(compressed_index_array_data),
            "ascii_array_size": sys.getsizeof(ascii_array_data),
            "gzipped_ascii_array_size": len(compressed_ascii_array_data),
            "original_ascii_size": original_ascii_size,
            "original_image_size": len(original_image_data),
            "gzipped_original_image_size": len(compressed_image_data),
            "gzipped_original_ascii_size": gzipped_ascii_size
        }

        # Print the stats for debugging
        print(f"Stats for image index {original_idx}: {original_images_data[original_idx]['stats']}")

    return original_images_data

from PIL import ImageSequence

def split_and_process_rgb(image, display_image_func, config, process_parameters):
    """ Split the image into RGB components, process each with display_image_func, and blend. Handle GIF by processing each frame. """
    
    # Check if the image is animated (contains multiple frames)
    if hasattr(image, "is_animated") and image.is_animated:
        frames = []
        for frame in ImageSequence.Iterator(image):
            frame = frame.copy()
            processed_frame = process_frame(frame, display_image_func, config, process_parameters)
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
        return process_frame(image, display_image_func, config, process_parameters)
import re
def process_frame(frame, display_image_func, config, process_parameters):
    complex_color = False
    """Process a single frame for RGB and optional alpha handling."""
    num_channels = len(frame.getbands())
    channel_names = ['red', 'green', 'blue']
    if num_channels == 1:
        channels = [frame]
    elif num_channels == 3:
        channels = frame.split()
    elif num_channels == 4:
        r, g, b, a = frame.split()
        channels = [r, g, b]
        alpha = a  # Preserve alpha channel for later recombination
    if process_parameters['invert']:
        for img in channels:
            img = ImageOps.invert(img)
    server_settings = process_parameters.get("server_settings", {})  # Get server-specific settings
    channel_settings = process_parameters["channel_settings"]
    channel_mode = process_parameters["channel_mode"]
    processed_channels = []
    if server_settings:

        if channel_mode == "hue_band":
            print("hue mode")
            channels_and_colors = create_hue_based_channels(frame, collect_hues_from_channel_settings(channel_settings))
            channels = list(channels_and_colors.values())
            print(f"channels: {channels}")
            colors = list(channels_and_colors.keys())
            print(f"colors: {colors}")
            processed_channels = [
                display_image_func(
                    config,
                    [ch],
                    True,
                    channel_setting
                )[0].convert("L")
                for ch, color, channel_setting in zip(channels, colors, channel_settings)]
            
            num_channels = len(colors)
            complex_color = True

        else:
            print(f"Wer're about to check for server settings in : {server_settings.items()}")
            # Example: Apply settings to each channel

            
            for (idx, channel), channel_setting in zip(enumerate(channels), channel_settings):
                print(channel_setting)
                # Updated display_image_func call using channel_settings_stripped
                processed_channel = display_image_func(
                            config,
                            [channel],
                            True,
                            channel_setting  # Pass the stripped settings dictionary
                        )[0].convert("L")
                processed_channels.append(processed_channel)
    else:
        # Proceed with original processing if no server-specific settings
        processed_channels = [
            display_image_func(
                config,
                [ch],
                True,
                {"h_offset":0, "v_offset":0, "rotation": 0, "r_oversampling":process_parameters["rotation_oversampling"],
                    "v_oversampling":process_parameters["vertical_oversampling"],
                    "h_oversampling":process_parameters["horizontal_oversampling"]}
            )[0].convert("L")
            for ch in channels
        ]
    # Proceed as originally defined without advanced hue-based processing
    #processed_channels = [display_image_func(config, [ch], True, process_parameters['rotation_oversampling'], process_parameters['vertical_oversampling'], process_parameters['horizontal_oversampling'])[0].convert('L') for ch in channels]

    for i, ch in enumerate(processed_channels):
        processed_channels[i] = ch.resize(processed_channels[0].size, Image.LANCZOS)    

    if not complex_color:
        if num_channels == 4:
            alpha_resized = alpha.resize(processed_channels[0].size, Image.LANCZOS)
            processed_channels.append(alpha_resized)
            return Image.merge('RGBA', tuple(processed_channels))
        elif num_channels == 3:
            return Image.merge('RGB', tuple(processed_channels))
        elif num_channels == 1:
            return processed_channels[0]
        
    else:
        for i in range(0, len(channel_settings)):
            channel_settings[i]["input_hue"]=channel_settings[i]["output_hue"]
        return_colors = collect_hues_from_channel_settings(channel_settings)
        return complex_color_merge(return_colors, processed_channels)

def create_hue_based_channels(frame, hues):
    return splitbyhues(frame, hues)

import colorsys

def collect_hues_from_channel_settings(channel_settings):
    def hex_to_rgb(hex_value):
        print(hex_value)
        hex_value = hex_value.lstrip('#')
        length = len(hex_value)
        return tuple(int(hex_value[i:i+length//3], 16) for i in range(0, length, length//3))
    
    def rgb_to_hue(rgb):
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        print(h*360)
        return h*360

    hues = {}
    for channel in channel_settings:
        hues.update({channel['input_hue']:rgb_to_hue(hex_to_rgb(channel['input_hue']))})

    print(f"hues: {hues}")
    return hues

# Placeholder for custom hue merging function
def custom_hue_merge_function(channels):
    # Implement your merging logic here
    # For demonstration, simply combine channels by averaging
    combined = sum(channels) // len(channels)
    return Image.fromarray(combined, 'L')

def process_image(img, process_parameters):
    with torch.no_grad():
        # Load your model and any other initial setup
        load_model(default_config)
        image_format = 'PNG'
        # Process the image (assuming split_and_process_rgb has been adapted for animation)
        processed_img = split_and_process_rgb(img, display_image, default_config, process_parameters)

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
        torch.cuda.empty_cache()
        img_buffer.seek(0)
        data = {
            'image_data': base64.b64encode(img_buffer.getvalue()).decode("utf-8"),  # Encode image bytes as base64
            'image_format': image_format
        }

        return json.dumps(data).encode('utf-8')  # Encode response as bytes

class ChannelSettings:
    def __init__(self, type='default', enable=False, bypass=False, alpha_multiplier=1, rotation=0, h_offset=0, v_offset=0, h_oversampling=0, r_oversampling=0, v_oversampling=0, input_hue=0, output_hue=0):
        self.type = type
        self.enable = enable
        self.bypass= bypass
        self.alpha_multiplier = alpha_multiplier
        self.rotation = rotation
        self.horizontal_oversampling = h_oversampling
        self.vertical_oversampling = v_oversampling
        self.rotational_oversampling = r_oversampling
        self.input_hue = input_hue
        self.output_hue = output_hue


def process_form(form_data):
    form_data = json.loads(form_data.get('settings_json'))
    process_parameters = {
        "rotation_oversampling": int(form_data.get("rotation_oversampling", 0)),
        "vertical_oversampling": int(form_data.get("vertical_oversampling", 0)),
        "horizontal_oversampling": int(form_data.get("horizontal_oversampling", 0)),
        "resize_width": int(form_data.get("resize_width", 0)),
        "resize_height": int(form_data.get("resize_height", 0)),
        "use_all_channels": form_data.get("use_all_channels", "on").lower() == "on",
        "convert_rgb_to_l": form_data.get("convert_rgb_to_l", "off").lower() == "on",
        'invert': form_data.get("invert", "off").lower() == "on",
        
    }

    bypass_mode = form_data.get('bypass_mode', False)
    convert_to_greyscale = form_data.get('convert_to_greyscale', False)
    resize_width = form_data.get('resize_width', 0)
    resize_height = form_data.get('resize_height', 0)
    server_name = form_data.get('name', '')
    color_settings = form_data.get('color', {})
    print(server_name)
    settings_type = color_settings.get(f'{server_name}_settings_type', 'error') # hue_band, rgb
    print(settings_type)
    settings_prefix = f"{settings_type}_"
    channel_settings = []
    channels = []
    print(color_settings)
    print(f"Settings type: {settings_type}")
    if settings_type == 'error':
        exit()
    if settings_type == "rgb" or settings_type == "default":
        channels = ['red', 'green', 'blue']
    elif settings_type == "hue_band":
        pattern = re.compile(f"^{settings_prefix}(\\d+)_enabled$")
        for key in color_settings:
            match = pattern.match(key)    
            if match:
                if color_settings.get(key, False):
                    print("match")
                    channels.append(int(match.group(1)))
            else:
                print(f"no match: {key}")

    process_parameters["channel_mode"] = settings_type

    for channel in channels:
        channel_prefix = ""
        if settings_type == "default":
            channel_prefix = f"{settings_prefix}"
        else:
            channel_prefix = f"{settings_prefix}{channel}_"
        channel_settings.append({
                                        "enabled": color_settings.get(f"{channel_prefix}enabled", False),
                                        "bypass": color_settings.get(f"{channel_prefix}bypass", False),
                                        "invert": color_settings.get(f"{channel_prefix}invert", False),
                                        "alpha_multiplier": color_settings.get(f"{channel_prefix}alpha_multiplier", 1),
                                        "hue_width": color_settings.get(f"{channel_prefix}hue_width", 0),
                                        "rotation": color_settings.get(f"{channel_prefix}rotation", 0),
                                        "h_offset": color_settings.get(f"{channel_prefix}h_offset", 0),
                                        "v_offset": color_settings.get(f"{channel_prefix}v_offset", 0),
                                        "h_oversampling": color_settings.get(f"{channel_prefix}h_oversampling", 0),
                                        "r_oversampling": color_settings.get(f"{channel_prefix}r_oversampling", 0),
                                        "v_oversampling": color_settings.get(f"{channel_prefix}v_oversampling", 0),
                                })

        if settings_type == "hue_band":
            channel_settings[-1]["input_hue"] = color_settings.get(f"{channel_prefix}input_hue_color", -1)
            channel_settings[-1]["output_hue"] = color_settings.get(f"{channel_prefix}output_hue_color", channel_settings[-1]["input_hue"])

    
    acceptable_keys = {key : form_data[key] for key in form_data if isinstance(key, str)}
    process_parameters['server_settings'] = acceptable_keys
    process_parameters['channel_settings'] = channel_settings
    print(process_parameters)
    return process_parameters

default_config.model_loaded = False
if load_model(default_config):
    default_config.model_loaded = True


# --- Encapsulate all Flask/server imports and code ---
if RUN_SERVER:
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    # Initialize Flask app and enable CORS
    app = Flask(__name__)
    CORS(app)

    # Check if the platform is Windows
    is_windows = platform.system().lower() == 'windows'

    if is_windows:
        from waitress import serve  # Import waitress conditionally

    @app.route('/process_image', methods=['POST'])
    def handle_image_processing():
        # Check if an image file is present in the request
        if 'image_file' not in request.files:
            error_message = "Image file not found in the request."
            print(error_message)  # Output error to stdout
            return jsonify({'error': error_message}), 400

        file = request.files['image_file']
        process_parameters = process_form(request.form)
        
        # Attempt to open the image using PIL

        max_pixels = 5000000
        img = Image.open(file.stream)
        if process_parameters['resize_width'] == 0:
            process_parameters['resize_width'] = img.size[0]
        if process_parameters['resize_height'] == 0:
            process_parameters['resize_height'] = img.size[1]
        img = img.resize((process_parameters['resize_width'], process_parameters['resize_height']))
        
        width, height = img.size
        aspect_ratio = width / height

        new_width = width
        new_height = height

        if width * height > max_pixels:
            new_width = int((max_pixels * aspect_ratio) ** .5)
            new_height = int(new_width / aspect_ratio)

            if new_width * new_height > max_pixels:
                new_height = int((max_pixels / aspect_ratio) ** .5)
                new_width = int(new_height * aspect_ratio)

            img = img.resize((new_width, new_height))

        if process_parameters['convert_rgb_to_l']:
            img = img.convert('L')
        processed_data = process_image(img, process_parameters)  # Replace with your actual image processing function

        # Ensure the processing function returns JSON data as a byte string
        return processed_data.decode('utf-8'), 200, {'Content-Type': 'application/json'}

        #except Exception as e:
        #    error_message = f"Error processing image: {str(e)}"
        #    print(request.files)
        #    print(error_message)  # Output the error to stdout
        #    return jsonify({'error': error_message}), 500

    if __name__ == '__main__':
        # If on Windows, use Waitress to serve the Flask app
        if is_windows:
            serve(app, host='0.0.0.0', port=PORT)
        else:
            # Use Flask's built-in server for other platforms
            app.run(host='0.0.0.0', port=PORT)

# --- Utility: ASCII text output function ---
def ascii_text_from_image(image_path, config=None):
    """Load an image, run through model, return ASCII text."""
    img = Image.open(image_path)
    if config is None:
        config = default_config
    ascii_text, _ = pilToASCII(img, config, max_size=img.size)
    return ascii_text

# --- If run as main with --text, output ASCII and exit ---
if __name__ == "__main__" and TEXT_ONLY and INPUT_IMAGE_PATH:
    ascii_text = ascii_text_from_image(INPUT_IMAGE_PATH)
    print(ascii_text)
    sys.exit(0)

# --- Expose main utility as a class for import ---
class FontMapperModel:
    def __init__(self, config=None):
        if config is None:
            self.config = default_config
        else:
            self.config = config

    def ascii_from_image(self, image_path):
        return ascii_text_from_image(image_path, self.config)

    def ascii_from_pil(self, pil_image):
        ascii_text, _ = pilToASCII(pil_image, self.config, max_size=pil_image.size)
        return ascii_text

    # Optionally expose more model/utility functions as needed