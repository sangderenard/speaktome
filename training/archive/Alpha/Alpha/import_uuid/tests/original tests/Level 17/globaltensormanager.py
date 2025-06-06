import torch
import hashlib
import time
import numpy as np
import math
import random
import torch.fft as fft

############################
# Error Correction (Basic)
############################

class ECC:
    @staticmethod
    def parity_bit(tensor):
        tensor_int = tensor.to(torch.int64)
        # Simple parity: XOR reduce
        parity = tensor_int.bitwise_xor(tensor_int >> 1).bitwise_and(1)
        return parity

    @staticmethod
    def hamming_encode(value):
        H = np.array([[1, 1, 0, 1],
                      [1, 0, 1, 1],
                      [0, 1, 1, 1]])
        value_bits = np.array([int(b) for b in f"{value:04b}"])
        parity_bits = H @ value_bits % 2
        return np.concatenate((value_bits, parity_bits))

    @staticmethod
    def hamming_decode(encoded):
        H_T = np.array([[1, 1, 0, 1],
                        [1, 0, 1, 1],
                        [0, 1, 1, 1]])
        syndrome = H_T @ encoded % 2
        error_position = int("".join(map(str, syndrome[::-1])), 2)
        if error_position > 0:
            encoded[error_position - 1] ^= 1
        return encoded[:4]

############################
# Dimension-Aware Repair
############################

class DimensionAwareRepair:
    @staticmethod
    def repair(tensor, scheme="mean"):
        # Replace NaNs with mean or zero
        if scheme == "mean":
            mean_values = tensor.nanmean()
            tensor = torch.nan_to_num(tensor, nan=mean_values.item())
        elif scheme == "zero":
            tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor

############################
# FFT-Based Salt Generation
############################

def generate_fft_salt(size, device='cpu'):
    """
    Generate a private salt by creating random frequency domain data,
    then inverse FFT to produce a time-domain salt vector.
    """
    # Create random frequency domain data (complex)
    freq_data = torch.randn(size, dtype=torch.complex64, device=device)
    # Inverse FFT to get time domain
    time_data = fft.ifft(freq_data)
    return time_data

def apply_salt_to_hash(hash_value, salt):
    """
    Combine hash_value (string) and salt (complex tensor) to produce a salted hash.
    We'll take an FFT of the salt again and combine its magnitude/phase with the hash.
    """
    # Convert salt to CPU numpy for stable hashing
    salt_cpu = salt.cpu().numpy()
    # Extract magnitude and phase from salt
    magnitude = np.abs(salt_cpu)
    phase = np.angle(salt_cpu)
    # Convert to bytes
    mag_bytes = magnitude.tobytes()
    phase_bytes = phase.tobytes()

    combined = (hash_value.encode('utf-8') + mag_bytes + phase_bytes)
    return hashlib.md5(combined).hexdigest()

############################
# Tensor Wrapper with ECC and FFT Salt
############################

class TensorWrapperECC_FFT:
    def __init__(self, tensor, name="tensor", device='cpu', repair_scheme="mean"):
        self.tensor = tensor.clone()
        self.name = name
        self.timestamp = time.time()
        self.parity = ECC.parity_bit(self.tensor)
        self.device = device
        self.repair_scheme = repair_scheme

        # Generate a private salt (FFT-based)
        # Let salt size be related to tensor size. For simplicity, length = product of dims
        length = self.tensor.numel()
        self.salt = generate_fft_salt(length, device=device)

        # Compute initial checksum with salt
        self.checksum = self._compute_salted_checksum()

    def _compute_salted_checksum(self):
        tensor_bytes = self.tensor.cpu().numpy().tobytes()
        base_hash = hashlib.md5(tensor_bytes).hexdigest()
        salted_hash = apply_salt_to_hash(base_hash, self.salt)
        return salted_hash

    def validate(self):
        # Check parity first
        current_parity = ECC.parity_bit(self.tensor)
        print("Original Parity:", self.parity)
        print("Current Parity:", current_parity)
        if not torch.equal(current_parity, self.parity):
            print("Parity Mismatch Detected!")
            return False
        
        # Check salted checksum
        current_hash = self._compute_salted_checksum()
        print("Stored Checksum:", self.checksum)
        print("Current Checksum:", current_hash)
        return current_hash == self.checksum


    def repair_if_needed(self, redundant_tensor=None):
        """
        Attempt repair if validation fails:
        1. Try dimension-aware repair (if it's NaN or small error)
        2. If provided redundant copy, use it
        """
        if self.validate():
            return True
        # Attempt dimension-aware repair
        self.tensor = DimensionAwareRepair.repair(self.tensor, scheme=self.repair_scheme)
        if self.validate():
            return True

        if redundant_tensor is not None:
            # Last resort: use redundant copy
            self.tensor = redundant_tensor.clone()
            # Update parity and checksum after repair
            self.parity = ECC.parity_bit(self.tensor)
            self.checksum = self._compute_salted_checksum()
            return self.validate()

        return False

    def refresh(self):
        # Refresh timestamp and re-hash
        self.timestamp = time.time()
        self.checksum = self._compute_salted_checksum()

    def base(self):
        return self.tensor

############################
# Global Tensor Manager
############################

class GlobalTensorManager:
    def __init__(self, float_precision=torch.float32, int_precision=torch.int32, device=None, requires_grad=False):
        self.float_precision = float_precision
        self.int_precision = int_precision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.requires_grad = requires_grad

    def to_tensor(self, data, tensor_type="float", requires_grad=None):
        requires_grad = self.requires_grad if requires_grad is None else requires_grad
        if tensor_type == "float":
            precision = self.float_precision
            tensor = torch.tensor(data, dtype=precision, device=self.device)
            if requires_grad and not tensor.requires_grad:
                tensor.requires_grad_()
        elif tensor_type == "int":
            precision = self.int_precision
            tensor = torch.tensor(data, dtype=precision, device=self.device)
        else:
            raise ValueError("tensor_type must be 'float' or 'int'")
        return tensor

    def _convert_to_float(self, tensor):
        if tensor.dtype not in [torch.float32, torch.float64]:
            tensor = tensor.to(dtype=self.float_precision, device=self.device)
        if self.requires_grad and not tensor.requires_grad:
            tensor = tensor.clone().requires_grad_()
        return tensor

    def ensure_mixed_type(self, tensorA, tensorB):
        tensorA = self._convert_to_float(tensorA)
        tensorB = self._convert_to_float(tensorB)
        return tensorA, tensorB

    def mixed_add(self, tensorA, tensorB):
        tensorA, tensorB = self.ensure_mixed_type(tensorA, tensorB)
        return tensorA + tensorB

    def mixed_multiply(self, tensorA, tensorB):
        tensorA, tensorB = self.ensure_mixed_type(tensorA, tensorB)
        return tensorA * tensorB

    def clone_and_track(self, tensor):
        tensor = tensor.clone()
        if self.requires_grad and not tensor.requires_grad:
            tensor.requires_grad_()
        return tensor

############################
# Demonstration Function
############################

def demonstration():
    device = "cpu"
    manager = GlobalTensorManager(float_precision=torch.float64, device=device, requires_grad=False)

    # Create a float tensor
    data = [[1.0, 2.0], [3.0, 4.0]]
    t = manager.to_tensor(data, tensor_type="float")

    # Wrap the tensor with ECC and FFT salt
    wrapper = TensorWrapperECC_FFT(t, name="demo_tensor", device=device)

    print("Original Validation:", wrapper.validate())

    # Introduce a corruption
    wrapper.tensor[0,0] = 99.99
    print("Validation After Corruption:", wrapper.validate())

    # Attempt dimension-aware repair
    repaired = wrapper.repair_if_needed()
    print("Validation After Repair Attempt:", repaired)

    # Introduce NaN to see dimension-aware repair in action
    wrapper.tensor[0,1] = float('nan')
    # Force re-validate
    print("Validation After Introducing NaN:", wrapper.validate())
    repaired = wrapper.repair_if_needed()
    print("Validation After Repairing NaN:", repaired)

    # Without correct FFT salt, verification should fail:
    # Simulate losing the salt by changing it
    original_salt = wrapper.salt.clone()
    wrapper.salt = wrapper.salt * 0.0 + 1.0  # corrupt salt
    print("Validation After Salt Corruption:", wrapper.validate())

    # Restore salt and check again
    wrapper.salt = original_salt
    wrapper.refresh()  # re-hash with correct salt
    print("Validation After Restoring Salt:", wrapper.validate())

    # Show mixed operations
    int_tensor = manager.to_tensor([[10,20],[30,40]], tensor_type="int")
    sum_result = manager.mixed_add(wrapper.base(), int_tensor)
    mul_result = manager.mixed_multiply(wrapper.base(), int_tensor)
    print("Mixed Add Result:", sum_result)
    print("Mixed Multiply Result:", mul_result)

if __name__ == "__main__":
    demonstration()
