#version 420 core
#define FOCAL_ATOMIC_MULTIPLIER .0001

#define MAX_PAIN 200
in vec2 texCoords; 
layout(binding = 7, offset = 0) uniform atomic_uint focalDistanceCounter;
layout(binding = 6) uniform sampler2D hdrBuffer;           // HDR color buffer
layout(binding = 5) uniform sampler2D oldBuffer;           // HDR color buffer
layout(binding = 3) uniform sampler2D distanceBuffer;      // Distance buffer

layout(binding = 1, offset = 0) uniform atomic_uint oldAperture;   // Atomic counter for old aperture
layout(binding = 2, offset = 0) uniform atomic_uint newAperture;   // Atomic counter for new aperture

layout(location = 1) out vec4 blurredColor;                // Output blurred HDR image
layout(location = 0) out vec4 focalClarity;

const float ATOMIC_MULTIPLIER = 0.00000001;
const float CLIPPING_THRESHOLD = 0.75;                     
const float PAIN_SCALE = 5.0;                              
const int NUM_SAMPLES = 16;

// Define an exaggeration coefficient to amplify the depth of field effect
const float EXAGGERATION_COEFFICIENT = .000001;  // Increase or decrease this to amplify or reduce the effect





float getFocalDistance() {
    float focalDistanceBase = float(atomicCounter(focalDistanceCounter));
    
    // Define the soft clamp threshold and scaling for non-linear response
    float threshold = 100000.0;
    float scaleFactor = 1.0;  // Controls the steepness of the approach
    
    float clampedExponent = clamp(-FOCAL_ATOMIC_MULTIPLIER * (focalDistanceBase - threshold), -2, 2);
    float microMode = max(0, scaleFactor * (1.0 - exp(clampedExponent)));
    focalDistanceBase = max(1.0, focalDistanceBase - 2000000000.0);  // Prevents negative apertureCount
    
    return max(.01, focalDistanceBase * FOCAL_ATOMIC_MULTIPLIER * microMode);
}
float getOldAperture() {
    float apertureCount = float(atomicCounter(oldAperture));
    
    // Define the soft clamp threshold and scaling for non-linear response
    float threshold = 10000000.0;
    float scaleFactor = 1.0;  // Controls the steepness of the approach
    
    float clampedExponent = clamp(-ATOMIC_MULTIPLIER * (apertureCount - threshold), -2.0, 2.0);
    float microMode = max(0, scaleFactor * (1.0 - exp(clampedExponent)));
    float adjustedAperture = max(1.0, apertureCount - 2000000000.0);

    return adjustedAperture * ATOMIC_MULTIPLIER * microMode;
}


// Adjust brightness based on aperture
float applyBrightnessAdjustment(vec3 color, float aperture) {
    float apertureAreaScale = aperture * aperture * 3.14159;
    vec3 scaledColor = color * apertureAreaScale;
    return dot(scaledColor, vec3(0.2126, 0.7152, 0.0722)); // Luminance calculation
}

// Calculate pain metric based on brightness
float calculatePain(float brightness) {
    return min(MAX_PAIN, pow(max(0.0, (brightness - CLIPPING_THRESHOLD) * 100.0), PAIN_SCALE));
}

// Calculate a non-linear blur radius based on spherical depth of field model
float calculateSphericalBlurRadius(float z) {
    float focalDepth = getFocalDistance();
    float depthDifference = abs(z - focalDepth);
    float radius = getOldAperture() * (depthDifference / (z * focalDepth));

    // Apply non-linear attenuation for more natural falloff and amplify by EXAGGERATION_COEFFICIENT
    return radius * EXAGGERATION_COEFFICIENT * (log(depthDifference + 1.0) / (1.0 + log(depthDifference + 1.0)));
}

// Apply circular aperture blur with depth-aware weighting
vec3 applyCircularApertureBlur(sampler2D buffer, vec2 uv, float blurRadius) {
    vec3 colorSum = vec3(0.0);
    float totalWeight = 0.0;

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        float angle = 2.0 * 3.14159265 * (float(i) / float(NUM_SAMPLES));  // Angle around circle
        float radialOffset = float(i % 3) / 3.0;  // Vary radius for inner, mid, and outer samples

        vec2 offset = vec2(cos(angle), sin(angle)) * blurRadius * radialOffset;
        vec3 sampleColor = texture(buffer, uv + offset).rgb;

        // Weight based on proximity to focal depth (favor in-focus points)
        float sampleDepth = texture(distanceBuffer, uv + offset).r;
        float focalDepth = getFocalDistance();
        float weight = max(0.000001, 1.0 - abs(sampleDepth - focalDepth) / focalDepth);

        colorSum += sampleColor * weight;
        totalWeight += weight;
    }

    return colorSum / max(totalWeight, 0.0001);  // Normalize by total weight
}

void main() {
    vec3 hdrColor = texture(hdrBuffer, texCoords).rgb;      // Sample HDR color
    vec3 oldColor = texture(oldBuffer, texCoords).rgb;
    float currentAperture = getOldAperture();               // Get current aperture value

    // Apply blur before using blurredColor for brightness measurement
    float distanceToViewer = texture(distanceBuffer, texCoords).r; // Get distance value
    float blurRadius = calculateSphericalBlurRadius(distanceToViewer);
    blurredColor = vec4(applyCircularApertureBlur(hdrBuffer, texCoords, blurRadius), texture(hdrBuffer, texCoords).a);

    // Measure brightness on the blurredColor instead of hdrColor
    float brightness = applyBrightnessAdjustment(oldColor, currentAperture);

    vec2 centerOffset = texCoords - vec2(0.5);
    float centerWeight = exp(-(centerOffset.x * centerOffset.x + centerOffset.y * centerOffset.y) * 2);

    float midtoneTarget = 0.5;
    float adjustment = (midtoneTarget - brightness) * centerWeight;

    float pain = calculatePain(brightness);

    focalClarity = vec4(blurRadius, pain, .5 - adjustment, centerWeight);

    // Adjust the aperture based on the brightness adjustment
    if (adjustment > 0.0 && pain < 1.0) {
        if (atomicCounter(newAperture) < 3900000000){
            atomicCounterIncrement(newAperture);
        }
    } else if (adjustment < 0.0) {
        for (uint i = 0; pow(i,2) < uint(pain); ++i) {
            if (atomicCounter(newAperture) > 100000) {
                atomicCounterDecrement(newAperture);
            }
        }
    }

}
