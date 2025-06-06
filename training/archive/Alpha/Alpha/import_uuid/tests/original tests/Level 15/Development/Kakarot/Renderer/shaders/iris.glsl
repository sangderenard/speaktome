#version 420 core

in vec2 texCoords;
out vec4 FragColor;

layout (binding = 4 )uniform sampler2D hdrBuffer;         // Original HDR color buffer
layout(binding = 1, offset = 0) uniform atomic_uint newApertureCounter; // Atomic counter for new aperture
const float ATOMIC_MULTIPLIER = 0.00000001;

float getCurrentAperture() {
    float apertureCount = float(atomicCounter(newApertureCounter));
    
    
    // Define the soft clamp threshold and scaling for non-linear response
    float threshold = 100000.0;
    float scaleFactor = 1.0;  // Controls the steepness of the approach
    
    float clampedExponent = clamp(-ATOMIC_MULTIPLIER * (apertureCount - threshold), -2.0, 2.0);
    float microMode = max(0, scaleFactor * (1.0 - exp(clampedExponent)));
    float adjustedAperture = max(1.0, apertureCount - 2000000000.0);
    
    return adjustedAperture * ATOMIC_MULTIPLIER * microMode;
}

// Function to apply brightness adjustment based on aperture
vec3 applyBrightnessAdjustment(vec3 hdrColor, float aperture) {
    float apertureAreaScale = aperture * aperture * 3.14159; // Calculate the aperture area scale
    return hdrColor * apertureAreaScale; // Scale the HDR color
}

void main() {
    // Retrieve HDR color and alpha from the original scene render
    vec4 hdrColor = texture(hdrBuffer, texCoords);
    
    // Get the current aperture scaling factor
    float currentAperture = getCurrentAperture();
    
    // Scale the output based on the current aperture
    vec3 adjustedColor = applyBrightnessAdjustment(hdrColor.rgb, currentAperture);
    
    // Retain the original alpha channel
    float outputAlpha = hdrColor.a;
    
    // Output the final color with retained alpha, clamped to valid [0, 1] range
    FragColor = vec4(clamp(adjustedColor, 0.0, 1.0), outputAlpha);
}
