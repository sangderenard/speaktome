# Detailed TODO: Dynamic Header Recognition and Construction

## Context
This task involves implementing a robust system for dynamic recognition, validation, and construction of headers in the SPEAKTOME project. The goal is to ensure headers are fully compliant with the canonical template, represented as a tree structure, and capable of being dynamically parsed, compared, and reconstructed.

## Objectives
1. **Dynamic Recognition**:
   - Parse headers using regex to identify atomic units.
   - Represent header components in a tree structure with parent-child relationships.

2. **Validation**:
   - Compare the parsed tree to the canonical template tree.
   - Log precise differences between the header and the template.

3. **Construction**:
   - Enable reconstruction of headers from the tree structure.
   - Support pretty-printing and markdown-based flowchart-style table depiction of the header tree.

## Steps

### 1. Define Regex Patterns
- **Task**: Create regex patterns for each atomic unit in the header.
- **Prototype**:
  ```python
  HEADER_START_REGEX = re.compile(r"^# --- BEGIN HEADER ---$")
  HEADER_END_REGEX = re.compile(r"^# --- END HEADER ---$")
  DOCSTRING_REGEX = re.compile(r"^"""(.*?)"""$")
  IMPORT_REGEX = re.compile(r"^import (.+)$")
  TRY_BLOCK_REGEX = re.compile(r"^try:$")
  EXCEPT_BLOCK_REGEX = re.compile(r"^except (.+):$")
  ```
- **Output**: A dictionary of regex patterns for all header components.

### 2. Implement Tree-Based Representation
- **Task**: Develop a data structure to represent the header as a tree.
- **Prototype**:
  ```python
  class HeaderNode:
      def __init__(self, name: str, content: str = "", children: list[HeaderNode] = None):
          self.name = name
          self.content = content
          self.children = children or []

      def add_child(self, child: HeaderNode):
          self.children.append(child)

      def __str__(self):
          return f"{self.name}: {self.content}"
  ```
- **Output**: A tree structure representing the header.

### 3. Write Comparison Logic
- **Task**: Compare the parsed header tree to the canonical template tree.
- **Prototype**:
  ```python
  def compare_trees(parsed_tree: HeaderNode, template_tree: HeaderNode) -> list[str]:
      differences = []
      if parsed_tree.name != template_tree.name:
          differences.append(f"Mismatch: {parsed_tree.name} != {template_tree.name}")
      for child in template_tree.children:
          if child.name not in [c.name for c in parsed_tree.children]:
              differences.append(f"Missing: {child.name}")
      return differences
  ```
- **Output**: A list of differences between the parsed and template trees.

### 4. Develop Pretty-Printing Functionality
- **Task**: Create a function to pretty-print the header tree as a markdown flowchart-style table.
- **Prototype**:
  ```python
  def pretty_print_tree(node: HeaderNode, level: int = 0) -> str:
      indent = "  " * level
      result = f"{indent}- {node.name}: {node.content}\n"
      for child in node.children:
          result += pretty_print_tree(child, level + 1)
      return result
  ```
- **Output**: A markdown string representing the header tree.

### 5. Integrate with `validate_headers.py`
- **Task**: Update `validate_headers.py` to use the new dynamic recognition and validation system.
- **Prototype**:
  ```python
  def validate_header(file_path: Path, template_tree: HeaderNode):
      with file_path.open("r") as f:
          content = f.read()
      parsed_tree = parse_header(content)
      differences = compare_trees(parsed_tree, template_tree)
      if differences:
          print(f"Header validation failed for {file_path}:")
          print("\n".join(differences))
  ```
- **Output**: Enhanced validation logic in `validate_headers.py`.

## Deliverables
1. Regex patterns for header components.
2. Tree-based representation of headers.
3. Comparison logic for parsed and template trees.
4. Pretty-printing functionality for header trees.
5. Updated `validate_headers.py` with dynamic recognition and validation.

## Notes
- Ensure all changes are thoroughly tested.
- Document the new system in `AGENTS.md` and `headers.md`.
- Collaborate with other agents to ensure seamless integration.

## Priority
This task is high priority and should be addressed immediately to ensure header consistency across the project.
