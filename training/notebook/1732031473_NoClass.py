import mitsuba as mi
mi.set_variant('scalar_spectral')  # Use spectral mode for high physical accuracy

def create_blackbody_scene():
    return {
        'type': 'scene',
        'integrator': {
            'type': 'path',  # Path tracing integrator
            'max_depth': 10
        },
        'camera': {
            'type': 'thinlens',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 5],  # Camera position
                target=[0, 0, 0],  # Look at the origin
                up=[0, 1, 0]       # Up direction
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'rfilter': {'type': 'box'}
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'aperture_radius': 0.05  # Aperture size for thin lens effects
        },
        # Small blackbody emitter next to the camera
        'small_blackbody': {
            'type': 'rectangle',  # Shape for the blackbody emitter
            'to_world': mi.ScalarTransform4f.translate([0.5, 0, 4.9]) @ mi.ScalarTransform4f.scale([0.1, 0.1, 1]),
            'emitter': {
                'type': 'blackbody',
                'temperature': 3000  # Blackbody temperature in Kelvins
            }
        },
        # Larger blackbody emitter slightly behind and to the right
        'large_blackbody': {
            'type': 'rectangle',  # Shape for the blackbody emitter
            'to_world': mi.ScalarTransform4f.translate([1.0, -0.5, 4.5]) @ mi.ScalarTransform4f.scale([0.5, 0.5, 1]),
            'emitter': {
                'type': 'blackbody',
                'temperature': 4500  # Blackbody temperature in Kelvins
            }
        }
    }

def render_scene(scene_dict, output_file='output.png'):
    # Load the scene
    scene = mi.load_dict(scene_dict)
    # Render the scene (produces a tensor with spectral data)
    image = mi.render(scene)

    # Convert to a Bitmap
    bitmap = mi.Bitmap(image)

    # Save as HDR (to preserve spectral data)
    output_hdr_file = output_file.replace('.png', '.exr')
    bitmap.write(output_hdr_file)
    print(f"Rendered spectral HDR image saved to {output_hdr_file}")

    # Save as PNG (optional RGB conversion for visualization)
    rgb_image = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
    rgb_image.write(output_file)
    print(f"Rendered RGB image saved to {output_file}")

def main():
    # Create the scene with the camera and blackbody emitters
    scene_dict = create_blackbody_scene()
    # Render and save the scene
    render_scene(scene_dict)

if __name__ == "__main__":
    main()
