#version 420 core
layout (location = 0) in vec3 position;
layout (location = 1) in float red;     
layout (location = 2) in float green;   
layout (location = 3) in float blue;    
layout (location = 4) in vec3 normal;   

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 viewPos; // Camera position in world space

out vec3 fragWorldPos;
out vec3 fragViewPos;
out vec4 fragScreenPos;
out vec4 fragColor;
out vec4 radialNormal; 
out vec3 fragNormal;
out vec3 viewDirection; // Direction from fragment to camera
out vec3 cameraForward; // Forward vector of camera in world space

void main() {
    vec3 normalWorld = normalize(mat3(transpose(inverse(model))) * normal);
    float fragColorDist = length(vec3(red, green, blue) - vec3(.5, .5, .5));
    fragColor = vec4(red, green, blue, 1.0);

    vec4 worldPosition = model * vec4(position + normalWorld * fragColorDist * .5, 1.0);
    fragWorldPos = vec3(worldPosition);

    vec4 viewPosition = view * worldPosition;
    fragViewPos = vec3(viewPosition);

    fragScreenPos = projection * viewPosition;


    
    vec3 normalView = normalize(mat3(view) * normalWorld);

    vec3 upVec = vec3(0.0, 1.0, 0.0);
    float thetaWorld = acos(dot(normalWorld, upVec));
    float phiWorld = atan(normalWorld.z, normalWorld.x);

    vec3 viewDirView = vec3(0.0, 0.0, -1.0);
    float thetaView = acos(dot(normalView, viewDirView));

    float phiView = atan(normalView.y, normalView.x);

    radialNormal = vec4(thetaView, phiView, thetaWorld, phiWorld);

    // Calculate view direction and forward vector
    viewDirection = normalize(viewPos - fragWorldPos);
    cameraForward = normalize(-vec3(view[2])); // Assuming view[2] is the negative z-axis in view space
    fragNormal = normalWorld;
    gl_Position = fragScreenPos;
}
