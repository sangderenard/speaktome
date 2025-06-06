import threading
import queue
import time

class TextInterpreter:
    """
    The base IO text interpreter for graph operations.
    Runs in its own thread to manage inputs without blocking graph operations.
    """
    def __init__(self, control_node):
        self.control_node = control_node
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False

    def start(self):
        """Starts the interpreter thread."""
        self.running = True
        threading.Thread(target=self._run_io_loop, daemon=True).start()
        print("Text Interpreter started. Enter commands to interact with the graph.")

    def stop(self):
        """Stops the interpreter."""
        self.running = False
        print("Text Interpreter shutting down...")

    def _run_io_loop(self):
        """Main IO loop for processing user inputs."""
        while self.running:
            try:
                user_input = input(">> ").strip()
                if user_input in ["exit", "quit"]:
                    self.stop()
                    break

                self._process_command(user_input)
            except Exception as e:
                print(f"Error: {e}")

    def _process_command(self, command):
        """Processes text commands and maps to graph operations."""
        tokens = command.split()
        if not tokens:
            return

        try:
            cmd = tokens[0].lower()

            if cmd == "add_edge":
                src, dst = int(tokens[1]), int(tokens[2])
                self.control_node.add_edge([[src, dst]], layer_name="default")
                print(f"Edge added: {src} -> {dst}")

            elif cmd == "remove_edge":
                src, dst = int(tokens[1]), int(tokens[2])
                self.control_node.remove_edge([[src, dst]], layer_name="default")
                print(f"Edge removed: {src} -> {dst}")

            elif cmd == "summary":
                self.control_node.display_summary()

            elif cmd == "add_layer":
                layer_name = tokens[1]
                self.control_node.add_layer(layer_name)
                print(f"Layer '{layer_name}' added.")

            elif cmd == "remove_layer":
                layer_name = tokens[1]
                self.control_node.remove_layer(layer_name)
                print(f"Layer '{layer_name}' removed.")

            elif cmd == "forward":
                self.control_node.execute_forward()
                print("Forward step executed.")

            else:
                print("Unknown command. Available: add_edge, remove_edge, add_layer, remove_layer, summary, forward.")
        except Exception as e:
            print(f"Command Error: {e}")
