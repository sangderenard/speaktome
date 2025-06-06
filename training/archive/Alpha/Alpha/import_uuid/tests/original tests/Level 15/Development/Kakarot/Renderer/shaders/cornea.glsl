#version 420 core
layout (location = 0) in vec2 position;
out vec2 texCoords;

void main() {
    texCoords = (position + 1.0) / 2.0; // Map to [0, 1] range for texture sampling
    gl_Position = vec4(position, 0.0, 1.0);
}