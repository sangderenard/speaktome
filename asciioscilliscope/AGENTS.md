# ASCII Oscilloscope Guidance

The header files under `include/asciioscilliscope` serve as living design docs.
Do **not** trim their comments. Implementation files mirror the declared
interfaces with stubbed logic so that everything builds. Agents updating this
code should preserve the comment blocks and expand the stubs rather than
rewrite them from scratch.

The Eigen submodule must be initialized before building:

```bash
git submodule update --init eigen
```
