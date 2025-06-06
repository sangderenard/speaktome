# renderer/counter_translator.py

import math

class CounterTranslator:
    """
    A static class that provides methods to calculate focal distance and old aperture
    based on atomic counters.
    """

    # Constants as defined in your code
    FOCAL_ATOMIC_MULTIPLIER = .0001
    ATOMIC_MULTIPLIER = 0.00000001
    MAX_PAIN = 1000
    CLIPPING_THRESHOLD = 0.75
    PAIN_SCALE = 5.0
    NUM_SAMPLES = 16
    EXAGGERATION_COEFFICIENT = 1.0  # Exaggeration coefficient to amplify the depth of field effect
    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))
    @staticmethod
    def get_focal_distance(focal_distance_counter: int) -> float:
        """
        Calculate the focal distance based on the focalDistanceCounter atomic value.

        Args:
            focal_distance_counter (int): The value from the focalDistanceCounter atomic counter.

        Returns:
            float: The calculated focal distance.
        """
        focal_distance_base = focalDistanceBase = float(focal_distance_counter)

        # Define the soft clamp threshold and scaling for non-linear response
        threshold = 100_000_000.0
        scale_factor = 1.0  # Controls the steepness of the approach

        # Calculate microMode with non-linear scaling
        clampedExponent = CounterTranslator.clamp(-CounterTranslator.FOCAL_ATOMIC_MULTIPLIER * (focalDistanceBase - threshold), -20, 20)
        micro_mode = max(0, scale_factor * (1.0 - math.exp(clampedExponent)))
        focalDistanceBase = max(1.0, focalDistanceBase - 2000000000.0)

        # Prevents negative apertureCount by subtracting 2,000,000,000.0 and ensuring a minimum of 1.0
        focal_distance_base = max(1.0, focal_distance_base - 2_000_000_000.0)

        # Calculate and return the final focal distance
        focal_distance = focal_distance_base * CounterTranslator.FOCAL_ATOMIC_MULTIPLIER * micro_mode

        return focal_distance

    @staticmethod
    def get_aperture(aperture_counter: int) -> float:
        """
        Calculate the old aperture based on the oldAperture atomic value.

        Args:
            old_aperture_counter (int): The value from the oldAperture atomic counter.

        Returns:
            float: The calculated old aperture.
        """
        aperture_count = float(aperture_counter)

        # Define the soft clamp threshold and scaling for non-linear response
        threshold = 100_000_000.0
        scale_factor = 1.0  # Controls the steepness of the approach

        # Calculate microMode with non-linear scaling
        exponent = CounterTranslator.clamp(-CounterTranslator.ATOMIC_MULTIPLIER * (aperture_count - threshold), -20, 20)
        micro_mode = max(0.0, scale_factor * (1.0 - math.exp(exponent)))

        # Prevents negative apertureCount by subtracting 2,000,000,000.0 and ensuring a minimum of 1.0
        adjusted_aperture = max(1.0, aperture_count - 2_000_000_000.0)

        # Calculate and return the final old aperture
        old_aperture = adjusted_aperture * CounterTranslator.ATOMIC_MULTIPLIER * micro_mode

        return old_aperture