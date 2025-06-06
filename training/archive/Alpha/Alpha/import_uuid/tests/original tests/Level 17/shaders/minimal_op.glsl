#version 420 core
layout (local_size_x = 128) in;

layout (std430, binding = 0) buffer FleetBuffer {
    vec4 fleet[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < fleet.length()) {
        fleet[idx].xyz += vec3(1.0, 1.0, 1.0);  // Simple increment operation
    }
}
