import pygame
import yaml
import json
import copy

class HUDManager:
    DEFAULT_CONFIG = {  # Store the default HUD configuration in memory
        "hud_elements": [
            {"name": "vertex_info", "type": "label", "position": [10, 10], "size": [200, 30],
             "source": "get_vertex_stats", "style": {"font": "Arial", "font_size": 16, "color": "#00FF00"}},
            {"name": "force_equilibrium", "type": "gauge", "position": [250, 10], "size": [300, 20],
             "source": "get_equilibrium_percentage", "style": {"color": "#FFAA00"}},
            {"name": "layer_switcher", "type": "toggle", "position": [600, 10], "size": [150, 30],
             "options": ["Layer 1", "Layer 2", "Layer 3"], "callback": "switch_layer"}
        ]
    }

    def __init__(self, config_file=None):
        self.hud_elements = []
        self.config_data = copy.deepcopy(self.DEFAULT_CONFIG)  # Start with a fresh copy of default
        self.config_file = config_file
        if config_file:
            self.load_config(config_file)
        self.init_elements()

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith(('.yaml', '.yml')):
                    self.config_data = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    self.config_data = json.load(f)
            self.init_elements()
        except Exception as e:
            print(f"Error loading HUD config: {e}")

    def save_config(self, output_file):
        try:
            with open(output_file, 'w') as f:
                if output_file.endswith(('.yaml', '.yml')):
                    yaml.dump(self.config_data, f)
                elif output_file.endswith('.json'):
                    json.dump(self.config_data, f, indent=4)
        except Exception as e:
            print(f"Error saving HUD config: {e}")

    def reset_to_default(self):
        """Reset HUD to default configuration."""
        self.config_data = copy.deepcopy(self.DEFAULT_CONFIG)
        self.init_elements()

    def init_elements(self):
        self.hud_elements = []
        for elem_config in self.config_data.get('hud_elements', []):
            element = HUDElementFactory.create(elem_config)
            if element:
                self.hud_elements.append(element)

    def update(self, data_sources):
        for element in self.hud_elements:
            element.update(data_sources)

    def render(self, screen):
        for element in self.hud_elements:
            element.render(screen)

    def open_config_editor(self, screen):
        editor = ConfigEditor(self.config_data)
        editor.run(screen)
        self.init_elements()  # Reinitialize after editing

class ConfigEditor:
    def __init__(self, config_data):
        self.config_data = config_data
        self.running = True
        self.font = pygame.font.SysFont('Arial', 20)
        self.edited_data = self.flatten_config(self.config_data)

    def flatten_config(self, data, prefix=''):
        """Flatten nested dictionaries for key-value editing."""
        flat = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self.flatten_config(value, new_key))
            else:
                flat[new_key] = value
        return flat

    def update_config(self):
        """Rebuild nested dictionary from flattened data."""
        nested = {}
        for key, value in self.edited_data.items():
            keys = key.split('.')
            d = nested
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        return nested

    def run(self, screen):
        clock = pygame.time.Clock()
        input_box = pygame.Rect(50, 50, 400, 30)
        active_key = None

        while self.running:
            screen.fill((30, 30, 30))
            y_offset = 50

            for key, value in self.edited_data.items():
                label_surface = self.font.render(f"{key}: {value}", True, pygame.Color("white"))
                screen.blit(label_surface, (50, y_offset))
                y_offset += 40

            pygame.display.flip()
            clock.tick(30)

        self.config_data.clear()
        self.config_data.update(self.update_config())
# Factory for HUD Elements
class HUDElementFactory:
    @staticmethod
    def create(config):
        """Create HUD elements dynamically."""
        element_type = config.get('type', 'label')
        type_map = {
            "label": LabelElement,
            "gauge": GaugeElement,
            "toggle": ToggleElement,
        }
        return type_map.get(element_type, LabelElement)(config)

# Base HUD Element Class
class BaseHUDElement:
    def __init__(self, config):
        self.name = config.get('name', 'unnamed')
        self.position = tuple(config.get('position', [0, 0]))
        self.size = tuple(config.get('size', [100, 30]))
        self.style = config.get('style', {})
        self.source = config.get('source', None)  # Data hook

    def update(self, data_sources):
        """Update the element dynamically. Override in subclasses."""
        pass

    def render(self, screen):
        """Render the element to the screen. Override in subclasses."""
        pass

# Example HUD Elements
class LabelElement(BaseHUDElement):
    def __init__(self, config):
        super().__init__(config)
        self.text = config.get('default_text', 'Label')

    def update(self, data_sources):
        if self.source and self.source in data_sources:
            self.text = data_sources[self.source]()

    def render(self, screen):
        font = pygame.font.SysFont(self.style.get('font', 'Arial'), self.style.get('font_size', 20))
        text_surface = font.render(self.text, True, pygame.Color(self.style.get('color', '#FFFFFF')))
        screen.blit(text_surface, self.position)

class GaugeElement(BaseHUDElement):
    def __init__(self, config):
        super().__init__(config)
        self.value = 0

    def update(self, data_sources):
        if self.source and self.source in data_sources:
            self.value = data_sources[self.source]()

    def render(self, screen):
        pygame.draw.rect(screen, pygame.Color("#555555"), (*self.position, *self.size))
        fill_width = self.size[0] * self.value
        pygame.draw.rect(screen, pygame.Color(self.style.get('color', '#00FF00')), (*self.position, fill_width, self.size[1]))

class ToggleElement(BaseHUDElement):
    def __init__(self, config):
        super().__init__(config)
        self.options = config.get('options', [])
        self.current_index = 0
        self.callback = config.get('callback', None)

    def handle_input(self):
        self.current_index = (self.current_index + 1) % len(self.options)
        if self.callback:
            self.callback(self.options[self.current_index])

    def render(self, screen):
        font = pygame.font.SysFont(self.style.get('font', 'Arial'), self.style.get('font_size', 20))
        text_surface = font.render(f"{self.name}: {self.options[self.current_index]}", True, pygame.Color('#FFFFFF'))
        screen.blit(text_surface, self.position)

class BaseTool:
    """Abstract base class for tools."""
    def __init__(self, name):
        self.name = name

    def handle_event(self, event, visualizer):
        """Handle user input events like mouse clicks or key presses."""
        pass

    def update(self, visualizer):
        """Optional: Update the state of the tool each frame."""
        pass

    def draw_overlay(self, screen, visualizer):
        """Optional: Draw additional tool overlays."""
        pass

class VertexEditorTool(BaseTool):
    def __init__(self):
        super().__init__("Vertex Editor")
        self.dragging = False
        self.selected_vertex = None
        self.previous_mouse_pos = None

    def handle_event(self, event, visualizer):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left-click
            mouse_pos = pygame.mouse.get_pos()
            vertex_idx = visualizer.get_clicked_vertex(
                visualizer.map_to_screen(visualizer.rotate_vertices()[:, :2]), mouse_pos
            )
            if vertex_idx is not None:
                self.selected_vertex = vertex_idx
                self.dragging = True
                self.previous_mouse_pos = mouse_pos
                visualizer.move_history.append(visualizer.vertices.clone())  # Save for undo

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # Release
            self.dragging = False
            self.selected_vertex = None

    def update(self, visualizer):
        if self.dragging and self.previous_mouse_pos:
            current_mouse_pos = pygame.mouse.get_pos()
            dx = (current_mouse_pos[0] - self.previous_mouse_pos[0]) / visualizer.screen_size[0] * visualizer.xy_range[0]
            dy = (current_mouse_pos[1] - self.previous_mouse_pos[1]) / visualizer.screen_size[1] * visualizer.xy_range[1]
            self.previous_mouse_pos = current_mouse_pos

            movement_3d = visualizer.project_back_to_3d(dx, dy)
            if self.selected_vertex is not None:
                visualizer.vertices[self.selected_vertex] += movement_3d
                visualizer.metanetwork.data.pos = visualizer.vertices

class EdgeEditorTool(BaseTool):
    def __init__(self):
        super().__init__("Edge Editor")
        self.selected_vertices = []

    def handle_event(self, event, visualizer):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left-click
            mouse_pos = pygame.mouse.get_pos()
            vertex_idx = visualizer.get_clicked_vertex(
                visualizer.map_to_screen(visualizer.rotate_vertices()[:, :2]), mouse_pos
            )

            if vertex_idx is not None:
                if vertex_idx in self.selected_vertices:
                    self.selected_vertices.remove(vertex_idx)
                else:
                    self.selected_vertices.append(vertex_idx)

            if len(self.selected_vertices) == 2:
                edge = tuple(sorted(self.selected_vertices))
                if visualizer.metanetwork.edge_exists(edge):
                    visualizer.metanetwork.remove_edge(edge)
                else:
                    visualizer.metanetwork.add_edge(edge)
                visualizer.edges = visualizer.metanetwork.edges.t().tolist()
                self.selected_vertices = []

    def draw_overlay(self, screen, visualizer):
        """Highlight selected vertices."""
        screen_vertices = visualizer.map_to_screen(visualizer.rotate_vertices()[:, :2])
        for idx in self.selected_vertices:
            pygame.draw.circle(screen, (255, 255, 0), (int(screen_vertices[idx][0]), int(screen_vertices[idx][1])), 10, 2)

class ToolManager:
    def __init__(self):
        self.tools = {}
        self.active_tool = None

    def register_tool(self, tool):
        self.tools[tool.name] = tool
        if self.active_tool is None:
            self.active_tool = tool  # Set the first registered tool as active

    def switch_tool(self, tool_name):
        if tool_name in self.tools:
            self.active_tool = self.tools[tool_name]

    def handle_event(self, event, visualizer):
        if self.active_tool:
            self.active_tool.handle_event(event, visualizer)

    def update(self, visualizer):
        if self.active_tool:
            self.active_tool.update(visualizer)

    def draw_overlay(self, screen, visualizer):
        if self.active_tool:
            self.active_tool.draw_overlay(screen, visualizer)
