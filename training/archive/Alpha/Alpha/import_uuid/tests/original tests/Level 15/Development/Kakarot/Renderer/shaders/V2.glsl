#version 420 core

layout(binding = 0) uniform sampler2D hdrBuffer;
layout(binding = 1) uniform sampler2D distanceBuffer;
layout(location = 0) out vec4 interestScoreOutput;  // RGBA output

#define MAX_CILIARY_RESPONSE 200000000.0
#define MIN_DISTANCE 0.01
const int KERNEL_SIZE = 4;

const float DEPTH_EDGE_WEIGHT = 0.6;
const float CONTRAST_WEIGHT = 0.4;

void main() {
    ivec2 size = textureSize(hdrBuffer, 0);
    vec2 uv = gl_FragCoord.xy / vec2(size);  // Calculate UV from fragment coordinates
    vec3 centerColor = texture(hdrBuffer, uv).rgb;

    float maxContrast = 0.0;
    float largestDepthEdge = 0.0;  // Initialize to infinity
    float selectedDistance = 0.0;

    // Kernel processing around the pixel
    for (int dx = -KERNEL_SIZE; dx <= KERNEL_SIZE; ++dx) {
        for (int dy = -KERNEL_SIZE; dy <= KERNEL_SIZE; ++dy) {
            int nx = int(gl_FragCoord.x) + dx;
            int ny = int(gl_FragCoord.y) + dy;

            // Ensure we're within texture bounds
            if (nx < 0 || nx >= size.x || ny < 0 || ny >= size.y) continue;

            vec2 offsetUV = vec2(nx, ny) / vec2(size);
            vec3 neighborColor = texture(hdrBuffer, offsetUV).rgb;

            // Contrast calculation
            float contrast = length(neighborColor - centerColor);
            float distance = texture(distanceBuffer, offsetUV).r;

            // Update max contrast
            if (contrast > maxContrast) {
                maxContrast = contrast;
            }

            // Update closest depth edge
            float depthEdge = abs(distance - texture(distanceBuffer, uv).r);
            if (depthEdge > largestDepthEdge) {
                largestDepthEdge = depthEdge;
                selectedDistance = distance;
            }
        }
    }

    // Calculate individual components for output
    float contrastInterest = CONTRAST_WEIGHT * maxContrast;                // Red channel
    float distanceEdgeInterest = DEPTH_EDGE_WEIGHT * largestDepthEdge;     // Green channel
    float centerWeight = 1.0 / (length(uv - vec2(0.5))+1);                     // Blue channel

    // Store metrics in the output fragment
    interestScoreOutput = vec4(contrastInterest, distanceEdgeInterest, selectedDistance, centerWeight);
}
