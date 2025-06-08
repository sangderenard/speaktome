# Testing Scripts

This folder collects small, informal scripts that help you inspect parts of the project by hand. They are complementary to the automated `tests/` suite but serve a different purpose.

Use these utilities when you want to try things quickly without the full test harness.

- `lookahead_demo.py` showcases the lookahead controller interactively.
- `test_hub.py` runs `pytest` and writes `stub_todo.txt` so you can see which stub tests remain.
- `stub_todo.txt` is generated automatically and lists placeholders awaiting real tests.
- `benchmark_tensor_ops.py` measures basic operation speed across available tensor backends.

Feel free to add your own exploratory scripts here.
