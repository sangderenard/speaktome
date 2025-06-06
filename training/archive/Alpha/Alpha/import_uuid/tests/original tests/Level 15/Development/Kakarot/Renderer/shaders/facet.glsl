#version 420 core

in vec4 geoColor;
in vec3 geoNormal;
in vec3 geoWorldPos;
in vec3 geoViewPos;
in vec4 geoScreenPos;
in vec4 geoRadialNormal;
in vec3 geoViewDirection;
in vec3 geoCameraForward;
flat in vec3 vPositionA;
flat in vec3 vPositionB;
flat in vec3 vPositionC;

uniform int numLights;
#define MAX_LIGHTS 10

struct Light {
    vec3 position;
    vec3 direction;
    vec4 color;
};

uniform Light lights[MAX_LIGHTS];
uniform float shininess;
uniform float specularIntensity;
uniform vec4 ambientColor;
uniform vec3 viewPos;

layout (location = 0) out vec4 colorOut;
layout (location = 1) out vec4 distanceAndFovOut; // vec4(R,G,B,A): world, view, screen distance, and center weight angle
layout (location = 2) out vec4 normalAngleOut;    // special normal angle vector

void main() {
    float alpha = 1/(1 + (length(geoWorldPos-vPositionA)+length(geoWorldPos-vPositionB)+length(geoWorldPos-vPositionC)));
    vec4 baseColor = vec4( geoColor.rgb, geoColor.a * alpha);

    // Ambient lighting calculation
    vec4 ambient = ambientColor * baseColor;
    vec3 norm = normalize(geoNormal);
    vec4 result = ambient;

    // Lighting calculations (per-fragment precision)
    for (int i = 0; i < numLights; ++i) {
        vec3 lightDir = normalize(lights[i].position - geoWorldPos);
        
        // Diffuse shading calculation
        float diff = max(dot(norm, lightDir), 0.0);
        vec4 diffuse = diff * baseColor * lights[i].color;

        // Specular shading calculation
        vec3 halfwayDir = normalize(lightDir + geoViewDirection);
        float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
        vec4 specular = specularIntensity * spec * lights[i].color;

        result += diffuse + specular;
    }

    // Output final color
    colorOut = result;

    // Calculate distances in world, view, and screen space (these interpolated values maintain accuracy here)
    float worldDistance = length(geoWorldPos - viewPos);
    float viewDistance = length(geoViewPos - vec3(0.0, 0.0, 0.0)); // relative to view center
    
    vec3 ndcPos = geoScreenPos.xyz / geoScreenPos.w;
    //vec2 windowCoords = ((ndcPos.xy + vec2(1.0)) / 2.0) * viewportSize;
    
    float screenDistance = length(ndcPos.xy);//length(windowCoords - (viewportSize/ 2.0)); // relative to screen center

    // Calculate the angle in the field of view as the angle between the camera's forward direction and the fragment direction
    float centerWeightAngle = acos(dot(normalize(geoCameraForward), normalize(geoWorldPos - viewPos))); // Angle in radians

    distanceAndFovOut = vec4(worldDistance, viewDistance, screenDistance, centerWeightAngle);
    normalAngleOut = geoRadialNormal;
}
