class Shop:
    """
    The Shop class is the top-level manager for all aspects of the system. It
    coordinates workers, materials, tools, and machines while maintaining records
    and in-house research capabilities.
    """

    def __init__(self, name: str):
        self.name = name
        self.tools = {}
        self.materials = {}
        self.machines = {}
        self.workers = {}
        self.records = []
        self.research = {}

    # --- Core Functionality ---

    def add_tool(self, tool_name, tool_class):
        self.tools[tool_name] = tool_class

    def add_material(self, material_name, material_class):
        self.materials[material_name] = material_class

    def add_machine(self, machine_name, machine_class):
        self.machines[machine_name] = machine_class

    def add_worker(self, worker_name, worker_class):
        self.workers[worker_name] = worker_class

    def log(self, record: str):
        self.records.append(record)

    def conduct_research(self, topic: str, method):
        """
        Conduct in-house research and store findings.
        """
        self.research[topic] = method()

    def run(self):
        """
        Execute all workers and machines in the shop.
        """
        for worker in self.workers.values():
            worker.perform_task()

    # --- Display Functions ---

    def display_inventory(self):
        print(f"Shop: {self.name}")
        print("Tools:", list(self.tools.keys()))
        print("Materials:", list(self.materials.keys()))
        print("Machines:", list(self.machines.keys()))
        print("Workers:", list(self.workers.keys()))
        print("Records:", self.records)
        print("Research:", list(self.research.keys()))

# --- Tools ---
class Tool:
    """
    Base class for tools in the shop. Tools are used by machines or workers
    to perform specific tasks.
    """

    def __init__(self, name: str):
        self.name = name

    def use(self, *args, **kwargs):
        raise NotImplementedError("Tool 'use' method must be implemented by subclasses.")

class GlyphAssembler(Tool):
    def use(self, text: str):
        """
        Assembles glyphs into a tensor.
        """
        print(f"Assembling glyphs for text: {text}")
        # Glyph assembly logic here
        return f"Glyph tensor for '{text}'"

class GradientMaker(Tool):
    def use(self, dimensions: tuple):
        """
        Creates a gradient tensor.
        """
        print(f"Creating gradient for dimensions: {dimensions}")
        # Gradient creation logic here
        return f"Gradient tensor for dimensions {dimensions}"

# --- Materials ---
class Material:
    """
    Base class for materials in the shop.
    """

    def __init__(self, name: str, properties: dict):
        self.name = name
        self.properties = properties

    def describe(self):
        print(f"Material: {self.name}")
        print("Properties:", self.properties)

class Paper(Material):
    def __init__(self, name: str, properties: dict):
        super().__init__(name, properties)

class Ink(Material):
    def __init__(self, name: str, properties: dict):
        super().__init__(name, properties)

# --- Machines ---
class Machine:
    """
    Base class for machines in the shop.
    Machines handle tasks that are too complex or precise for workers alone.
    """

    def __init__(self, name: str, tools: list):
        self.name = name
        self.tools = tools

    def operate(self, *args, **kwargs):
        raise NotImplementedError("Machine 'operate' method must be implemented by subclasses.")

class GrandPress(Machine):
    def operate(self, glyph_tensor, gradient_tensor, mask_tensor, grand_sheet):
        """
        Apply glyphs, gradients, and masks to the grand sheet.
        """
        print(f"Operating Grand Press with {glyph_tensor}, {gradient_tensor}, {mask_tensor}, on {grand_sheet}")
        # Printing operation logic here
        return "Finished Grand Sheet"

class Digitizer(Machine):
    def operate(self, tensor):
        """
        Converts tensors into OpenGL textures.
        """
        print(f"Digitizing tensor: {tensor}")
        # OpenGL texture creation logic here
        return f"OpenGL Texture for {tensor}"

# --- Workers ---
class Worker:
    """
    Base class for workers in the shop. Workers use machines and tools to complete tasks.
    """

    def __init__(self, name: str, machines: list, tools: list):
        self.name = name
        self.machines = machines
        self.tools = tools

    def perform_task(self):
        raise NotImplementedError("Worker 'perform_task' method must be implemented by subclasses.")

class InkWorker(Worker):
    def perform_task(self):
        print(f"{self.name} is preparing ink.")
        # Ink preparation logic using tools and machines

class PaperWorker(Worker):
    def perform_task(self):
        print(f"{self.name} is preparing paper.")
        # Paper preparation logic using tools and machines

class PressOperator(Worker):
    def perform_task(self):
        print(f"{self.name} is operating the Grand Press.")
        # Press operation logic using tools and machines

# --- Demonstration ---
if __name__ == "__main__":
    # Create the shop
    shop = Shop("The Grand Print Shop")

    # Add tools
    shop.add_tool("Glyph Assembler", GlyphAssembler("Glyph Assembler"))
    shop.add_tool("Gradient Maker", GradientMaker("Gradient Maker"))

    # Add materials
    shop.add_material("Standard Paper", Paper("Standard Paper", {"weight": "80gsm", "color": "white"}))
    shop.add_material("Black Ink", Ink("Black Ink", {"viscosity": "medium", "color": "black"}))

    # Add machines
    shop.add_machine("Grand Press", GrandPress("Grand Press", ["Glyph Assembler", "Gradient Maker"]))
    shop.add_machine("Digitizer", Digitizer("Digitizer", []))

    # Add workers
    shop.add_worker("Alice", InkWorker("Alice", [], ["Glyph Assembler"]))
    shop.add_worker("Bob", PaperWorker("Bob", [], []))
    shop.add_worker("Charlie", PressOperator("Charlie", ["Grand Press"], []))

    # Conduct operations
    shop.display_inventory()
    shop.run()
