

import argparse
from graph import Graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Path to the system root directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--headless", action="store_true", help="Run without interpreter node")
    args = parser.parse_args()

    # Initialize the Graph
    graph = Graph(root_path=args.root, debug=args.debug, headless=args.headless)
    graph.run()
