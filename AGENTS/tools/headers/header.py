from typing import Any, Dict, List, Optional
import re

class Header:
    """A class to represent and manipulate headers dynamically."""

    def __init__(self, description: str, components: Optional[Dict[str, str]] = None, error_handling: Optional[Dict[str, str]] = None):
        self.description = description
        self.components = components or {}
        self.error_handling = error_handling or {}

    def __str__(self) -> str:
        """Pretty-print the header as a markdown string."""
        components_str = "\n".join([f"  - {key}: \"{value}\"" for key, value in self.components.items()])
        error_handling_str = "\n".join([f"  - {key}: \"{value}\"" for key, value in self.error_handling.items()])
        return f"""
HEADER:
  - Description: "{self.description}"
  - Components:
{components_str}
  - Error Handling:
{error_handling_str}
"""

    @classmethod
    def from_markdown(cls, markdown: str) -> "Header":
        """Parse a markdown template to create a Header object."""
        description_match = re.search(r"Description: \"(.*?)\"", markdown)
        components_match = re.findall(r"- (\w+): \"(.*?)\"", markdown.split("Components:")[1].split("Error Handling:")[0])
        error_handling_match = re.findall(r"- (\w+): \"(.*?)\"", markdown.split("Error Handling:")[1])

        description = description_match.group(1) if description_match else ""
        components = {key: value for key, value in components_match}
        error_handling = {key: value for key, value in error_handling_match}

        return cls(description, components, error_handling)

    def to_markdown(self) -> str:
        """Convert the Header object back to a markdown string."""
        return str(self)

    def interpret(self, instructions: List[str]) -> None:
        """Interpret procedural instructions for dynamic header manipulation."""
        for instruction in instructions:
            # Example: Add logic for GOTO-style procedural looping
            pass

# Example usage
if __name__ == "__main__":
    markdown_template = """
HEADER:
  - Description: "Template for SPEAKTOME module headers."
  - Components:
      - IMPORT_FAILURE_PREFIX: "Prefix for import failure messages."
      - auto_env_setup: "Module invoked via subprocess to initialize the environment."
      - ENV_SETUP_BOX: "Environment variable for setup box."
  - Error Handling:
      - KeyError: "Raised if ENV_SETUP_BOX is not set."
      - RuntimeError: "Raised if the environment is not initialized."
"""
    header = Header.from_markdown(markdown_template)
    print(header)
