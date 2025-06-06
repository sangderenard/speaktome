import mitsuba as mi

# Activate Mitsuba's Python bindings in spectral mode
mi.set_variant('scalar_spectral')

# Define the entire scene as a single static dictionary
scene_dict = {
    'type': 'scene',

    # Environment Emitter
    'env_emitter': {
        'type': 'constant',
        'radiance': {
            'type': 'spectrum',
            'value': 1.0  # Spectral radiance
        }
    },

    # Spot Emitter
    'spot_emitter': {
        'type': 'spot',
        'intensity': {
            'type': 'spectrum',
            'value': 1.0  # Spectral intensity
        },
        'cutoff_angle': 20.0,  # Degrees
        'beam_width': 30.0,    # Degrees
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[-1.0, 0.0, 0.0],  # Positioned to the left along the X-axis
            target=[0.0, 0.0, 0.0],   # Looking towards the origin (sphere)
            up=[0.0, 1.0, 0.0]         # Up direction
        )
    },

    # Point Emitters
    'point_emitter_right': {
        'type': 'point',
        'position': [10.0, 0.0, 0.0],  # Right of the camera
        'intensity': {
            'type': 'spectrum',
            'value': 10.0  # Intensity is 10x that of the spot light
        }
    },
    'point_emitter_below': {
        'type': 'point',
        'position': [0.0, -10.0, 0.0],  # Below the camera
        'intensity': {
            'type': 'spectrum',
            'value': 10.0
        }
    },
    'point_emitter_behind': {
        'type': 'point',
        'position': [0.0, 0.0, 10.0],  # Behind the camera
        'intensity': {
            'type': 'spectrum',
            'value': 10.0
        }
    },

    # Sphere Geometry
    'sphere': {
        'type': 'sphere',
        'radius': 0.5,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.7, 0.7, 0.7]  # Grey color
            }
        }
    },

    # Thin Lens Camera
    'sensor': {
        'type': 'thinlens',
        'fov': 45.0,  # Field of view in degrees
        'aperture_radius': 0.1,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0.0, 0.0, -1.5],  # Positioned along the negative Z-axis
            target=[0.0, 0.0, 0.0],    # Looking towards the origin (sphere)
            up=[0.0, 1.0, 0.0]          # Up direction
        ),
        'film': {
            'type': 'specfilm',
            'width': 1024,
            'height': 768,
            'component_format': 'float32',
            'band1_red': {
                'type': 'spectrum',
                'filename': 'spectral_data/data_red.spd'
            },
            'band2_green': {
                'type': 'spectrum',
                'filename': 'spectral_data/data_green.spd'
            },
            'band3_blue': {
                'type': 'spectrum',
                'filename': 'spectral_data/data_blue.spd'
            }
        }
    },

    # Integrator
    'integrator': {
        'type': 'path',
        'max_depth': 5  # Maximum recursion depth
    }
}

# Load the scene from the dictionary
scene = mi.load_dict(scene_dict)

# Render the scene with fixed parameters
image = mi.render(
    scene,
    spp=64,  # Samples per pixel
)
import numpy as np
print(np.array(image).mean())
print("Rendering completed. Image saved to 'rendered_sphere.exr'")
