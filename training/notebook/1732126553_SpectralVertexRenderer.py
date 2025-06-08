import mitsuba as mi
import numpy as np
import logging
import traceback

mi.set_variant('scalar_spectral')  # Use spectral mode for high physical accuracy

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpectralVertexRenderer:
    def __init__(self, camera_origin, camera_target, camera_up, film_width, film_height, aperture_radius):
        self.camera_origin = camera_origin
        self.camera_target = camera_target
        self.camera_up = camera_up
        self.film_width = film_width
        self.film_height = film_height
        self.aperture_radius = aperture_radius

    def normalize_vertices(self, vertices):
        try:
            vertices = np.array(vertices)
            min_coords = vertices.min(axis=0)
            max_coords = vertices.max(axis=0)
            scale = max(max_coords - min_coords)
            normalized_vertices = (vertices - min_coords) / scale
            logger.debug("Vertices normalized successfully.")
            return normalized_vertices.tolist()
        except Exception as e:
            logger.error("Error during vertex normalization: %s", e, exc_info=True)
            raise

    def vertices_to_obj(self, vertices, filename="subject.obj"):
        try:
            # Compute face normals and save vertices/normals to OBJ
            faces = [
                (0, 1, 2, 3),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (2, 3, 7, 6),
                (0, 3, 7, 4),
                (1, 2, 6, 5),
            ]
            normals = []
            for face in faces:
                p1 = np.array(vertices[face[1]]) - np.array(vertices[face[0]])
                p2 = np.array(vertices[face[2]]) - np.array(vertices[face[0]])
                normal = np.cross(p1, p2)
                normal /= np.linalg.norm(normal)
                normals.extend([normal] * 4)

            with open(filename, 'w') as f:
                for vertex in vertices:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                for normal in normals:
                    f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                for i, face in enumerate(faces):
                    indices = [idx + 1 for idx in face]
                    f.write(f"f {indices[0]}//{i + 1} {indices[1]}//{i + 1} {indices[2]}//{i + 1} {indices[3]}//{i + 1}\n")
            logger.debug("OBJ file written successfully: %s", filename)
            return filename
        except Exception as e:
            logger.error("Error while writing OBJ file: %s", e, exc_info=True)
            raise

    def create_scene(self, vertex_buffer, temperature, scale_normalization=False):
        try:
            if scale_normalization:
                vertex_buffer = self.normalize_vertices(vertex_buffer)

            obj_filename = self.vertices_to_obj(vertex_buffer)
            scene_dict = {
                'type': 'scene',
                'integrator': {
                    'type': 'path',
                    'max_depth': 20
                },
                'camera': {
                    'type': 'thinlens',
                    'to_world': mi.ScalarTransform4f.look_at(
                        origin=self.camera_origin,
                        target=self.camera_target,
                        up=self.camera_up
                    ),
                    'film': {
                        'type': 'hdrfilm',
                        'width': self.film_width,
                        'height': self.film_height,
                        'rfilter': {'type': 'box'}
                    },
                    'sampler': {
                        'type': 'independent',
                        'sample_count': 640
                    },
                    'aperture_radius': self.aperture_radius
                },
                'subject': {
                    'type': 'obj',
                    'filename': obj_filename,
                    'bsdf': {
                        'type': 'diffuse',
                        'reflectance': {'type': 'blackbody', 'temperature': temperature}
                    },
                    'emitter': {
                        'type': 'blackbody',
                        'temperature': 3000
                    }
                },
                'hot_light': {
                    'type': 'rectangle',
                    'to_world': mi.ScalarTransform4f.translate([0.5, 0, 4.9]) @ mi.ScalarTransform4f.scale([0.1, 0.1, 1]),
                    'emitter': {
                        'type': 'blackbody',
                        'temperature': 3000
                    }
                },
                'cool_light': {
                    'type': 'rectangle',
                    'to_world': mi.ScalarTransform4f.translate([2, -3, -5]) @ mi.ScalarTransform4f.scale([4, 4, 1]),
                    'emitter': {
                        'type': 'blackbody',
                        'temperature': 10000
                    }
                }
            }
            logger.debug("Scene dictionary created successfully.")
            return scene_dict
        except Exception as e:
            logger.error("Error while creating the scene: %s", e, exc_info=True)
            raise

    def render(self, vertex_buffer, temperature, scale_normalization=False, output_file="output.png"):
        try:
            scene_dict = self.create_scene(vertex_buffer, temperature, scale_normalization)
            scene = mi.load_dict(scene_dict)

            # Render the scene
            image = mi.render(scene)

            # Save the rendered image
            bitmap = mi.Bitmap(image)
            output_hdr_file = output_file.replace('.png', '.exr')
            bitmap.write(output_hdr_file)
            logger.info("Rendered spectral HDR image saved to %s", output_hdr_file)

            rgb_image = bitmap.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
            rgb_image.write(output_file)
            logger.info("Rendered RGB image saved to %s", output_file)
        except Exception as e:
            logger.error("Error during rendering: %s", e, exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    renderer = SpectralVertexRenderer(
        camera_origin=[0, 0, 5],
        camera_target=[0, 0, 0],
        camera_up=[0, 1, 0],
        film_width=512,
        film_height=512,
        aperture_radius=0.05
    )

    # Example cube vertex buffer
    cube_vertices = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]

    # Render the cube with a blackbody temperature of 4500K
    renderer.render(cube_vertices, temperature=3000, scale_normalization=True, output_file="output_cube.png")
