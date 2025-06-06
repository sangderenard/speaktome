#version 420 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 fragWorldPos[];
in vec3 fragViewPos[];
in vec4 fragScreenPos[];
in vec4 fragColor[];
in vec4 radialNormal[];
in vec3 viewDirection[];
in vec3 fragNormal[];
in vec3 cameraForward[];

out vec4 geoColor;
out vec3 geoNormal;
out vec3 geoWorldPos;
out vec3 geoViewPos;
out vec4 geoScreenPos;
out vec4 geoRadialNormal;
out vec3 geoViewDirection;
out vec3 geoCameraForward;
flat out vec3 vPositionA;
flat out vec3 vPositionB;
flat out vec3 vPositionC;

void main() {
    vPositionA = fragWorldPos[0];
    vPositionB = fragWorldPos[1];
    vPositionC = fragWorldPos[2];
    
    for (int i = 0; i < 3; i++) {
        geoColor = fragColor[i];
        geoNormal = fragNormal[i];
        geoWorldPos = fragWorldPos[i];
        geoViewPos = fragViewPos[i];
        geoScreenPos = fragScreenPos[i];

        geoRadialNormal = radialNormal[i];
        geoViewDirection = viewDirection[i];
        geoCameraForward = cameraForward[i];

        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
