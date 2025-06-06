import mitsuba as mi
mi.set_variant('scalar_spectral')
class MitsubaPlugin:
    def __init__(self, plugin_type, **params):
        """
        Initialize a Mitsuba plugin.
        
        :param plugin_type: The type of the plugin (e.g., 'sensor', 'emitter', 'shape', etc.).
        :param params: Key-value pairs of parameters for the plugin.
        """
        self.plugin_type = plugin_type
        self.params = params

    def to_dict(self):
        """
        Convert the plugin to a Mitsuba-compatible dictionary representation.
        
        :return: A dictionary representing the plugin.
        """
        plugin_dict = {"type": self.plugin_type}
        plugin_dict.update(self.params)
        return plugin_dict


class SceneBuilder:
    def __init__(self):
        """
        Initialize the scene builder with an empty list of plugins.
        """
        self.plugins = []

    def add_plugin(self, plugin):
        """
        Add a plugin to the scene.
        
        :param plugin: An instance of MitsubaPlugin.
        """
        if not isinstance(plugin, MitsubaPlugin):
            raise TypeError("Plugin must be an instance of MitsubaPlugin.")
        self.plugins.append(plugin)

    def build_scene(self):
        """
        Assemble the scene programmatically.
        
        :return: A Mitsuba scene dictionary.
        """
        scene_dict = {"type": "scene"}
        for idx, plugin in enumerate(self.plugins):
            scene_dict[f"plugin_{idx}"] = plugin.to_dict()
        return scene_dict


# Example Usage
def main():
    # Initialize the scene builder
    builder = SceneBuilder()

    # Add a sensor
    sensor_plugin = MitsubaPlugin(
        plugin_type="perspective",
        film={"type": "hdrfilm", "width": 1920, "height": 1080},
        sampler={"type": "independent", "sample_count": 64},
        fov=45
    )
    builder.add_plugin(sensor_plugin)

    # Add an emitter (light source)
    emitter_plugin = MitsubaPlugin(
        plugin_type="point",
        intensity={"type": "rgb", "value": [1.0, 1.0, 1.0]},
        position=[0, 5, 0]
    )
    builder.add_plugin(emitter_plugin)

    # Add a shape (sphere)
    shape_plugin = MitsubaPlugin(
        plugin_type="sphere",
        bsdf={"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.2, 0.2]}},
        
    )
    builder.add_plugin(shape_plugin)

    # Build the scene
    scene_dict = builder.build_scene()
    
    # Load the scene into Mitsuba
    scene = mi.load_dict(scene_dict)


if __name__ == "__main__":
    main()
