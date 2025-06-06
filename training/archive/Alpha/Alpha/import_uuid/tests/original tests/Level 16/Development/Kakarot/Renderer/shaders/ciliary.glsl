#version 420 core
#define FOCAL_ATOMIC_MULTIPLIER .0001
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 2) uniform sampler2D interestScoreBuffer;
layout(binding = 1, offset = 0) uniform atomic_uint focalDistanceCounter;

// Configurable importance weights
uniform float centerWeightImportance = 1.0;
uniform float distanceEdgeImportance = 1.0;
uniform float contrastImportance = 1.0;

// Maximum allowed focal distance change per update
const float MAX_CILIARY_RESPONSE = 200000000.0;

// Kernel size for maximum search
const int KERNEL_SIZE = 4;

float getFocalDistance() {
    float focalDistanceBase = float(atomicCounter(focalDistanceCounter));
    
    // Define the soft clamp threshold and scaling for non-linear response
    float threshold = 10000000.0;
    float scaleFactor = 1.0;  // Controls the steepness of the approach
    
    float clampedExponent = clamp(-FOCAL_ATOMIC_MULTIPLIER * (focalDistanceBase - threshold), -2, 2);
    float microMode = max(0, scaleFactor * (1.0 - exp(clampedExponent)));
    focalDistanceBase = max(1.0, focalDistanceBase - 2000000000.0);  // Prevents negative apertureCount
    
    return focalDistanceBase * FOCAL_ATOMIC_MULTIPLIER * microMode;
}


void adjustFocalDistance(float clampedChange) {

    for (int i = 0; i < min(abs(clampedChange), MAX_CILIARY_RESPONSE); ++i) {
        if (clampedChange > 0.0) {
            atomicCounterIncrement(focalDistanceCounter);
        } else if (clampedChange < 0.0) {
            atomicCounterDecrement(focalDistanceCounter);
        }
    }
}
void main() {
    ivec2 size = textureSize(interestScoreBuffer, 0);
    float maxScore = 0.0;
    float selectedDistance = 100.0; // Initialize with a default value

    // Iterate through interestScoreBuffer within the kernel boundaries
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            vec2 uv = vec2(x, y) / vec2(size);
            vec4 interestData = texture(interestScoreBuffer, uv);

            // Read individual interest components
            float contrastInterest = interestData.r;
            float distanceEdgeInterest = interestData.g;
            float centerWeight = interestData.a;
            float distance = interestData.b;

            // Calculate weighted score
            float score = (contrastInterest * contrastImportance) +
                          (distanceEdgeInterest * distanceEdgeImportance) +
                          (centerWeight * centerWeightImportance);

            // Update maximum score if this score is higher
            if (score > maxScore) {
                maxScore = score;
                selectedDistance = distance;
            }
        }
    }

    // Retrieve the current focal distance
    float currentFocalDistance = getFocalDistance();

    // Calculate the desired change in focal distance
    float distanceDifference = selectedDistance - currentFocalDistance;

    // Limit the change to within the maximum allowable step
    float clampedChange = clamp(distanceDifference, -MAX_CILIARY_RESPONSE,
                                MAX_CILIARY_RESPONSE);

    // Apply the incremental change through the atomic counter
    adjustFocalDistance(clampedChange);
}